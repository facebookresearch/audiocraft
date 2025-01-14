import torch
import typing as tp
from itertools import chain
from pathlib import Path
from torch import nn
from .conditioners import (ConditioningAttributes, BaseConditioner, ConditionType,
                           ConditioningProvider, JascoCondConst,
                           WaveformConditioner, WavCondition, SymbolicCondition)
from ..data.audio import audio_read
from ..data.audio_utils import convert_audio
from ..utils.autocast import TorchAutocast
from ..utils.cache import EmbeddingCache


class MelodyConditioner(BaseConditioner):
    """
    A conditioner that handles melody conditioning from pre-computed salience matrix.
    Attributes:
        card (int): The cardinality of the melody matrix.
        out_dim (int): The dimensionality of the output projection.
        device (Union[torch.device, str]): The device on which the embeddings are stored.
    """
    def __init__(self, card: int, out_dim: int, device: tp.Union[torch.device, str] = 'cpu', **kwargs):
        super().__init__(dim=card, output_dim=out_dim)
        self.device = device

    def tokenize(self, x: SymbolicCondition) -> SymbolicCondition:
        return SymbolicCondition(melody=x.melody.to(self.device))  # type: ignore

    def forward(self, x: SymbolicCondition) -> ConditionType:
        embeds = self.output_proj(x.melody.permute(0, 2, 1))  # type: ignore
        mask = torch.ones_like(embeds[..., 0])
        return embeds, mask


class ChordsEmbConditioner(BaseConditioner):
    """
    A conditioner that embeds chord symbols into a continuous vector space.
    Attributes:
        card (int): The cardinality of the chord vocabulary.
        out_dim (int): The dimensionality of the output embeddings.
        device (Union[torch.device, str]): The device on which the embeddings are stored.
    """
    def __init__(self, card: int, out_dim: int, device: tp.Union[torch.device, str] = 'cpu', **kwargs):
        vocab_size = card + 1  # card + 1 - for null chord used during dropout
        super().__init__(dim=vocab_size, output_dim=-1)  # out_dim=-1 to avoid another projection
        self.emb = nn.Embedding(vocab_size, out_dim, device=device)
        self.device = device

    def tokenize(self, x: SymbolicCondition) -> SymbolicCondition:
        return SymbolicCondition(frame_chords=x.frame_chords.to(self.device))   # type: ignore

    def forward(self, x: SymbolicCondition) -> ConditionType:
        embeds = self.emb(x.frame_chords)
        mask = torch.ones_like(embeds[..., 0])
        return embeds, mask


class DrumsConditioner(WaveformConditioner):
    def __init__(self, out_dim: int, sample_rate: int, blurring_factor: int = 3,
                 cache_path: tp.Optional[tp.Union[str, Path]] = None,
                 compression_model_latent_dim: int = 128,
                 compression_model_framerate: float = 50,
                 segment_duration: float = 10.0,
                 device: tp.Union[torch.device, str] = 'cpu',
                 **kwargs):
        """Drum condition conditioner

        Args:
            out_dim (int): _description_
            sample_rate (int): _description_
            blurring_factor (int, optional): _description_. Defaults to 3.
            cache_path (tp.Optional[tp.Union[str, Path]], optional): path to precomputed cache. Defaults to None.
            compression_model_latent_dim (int, optional): latent dimensino. Defaults to 128.
            compression_model_framerate (float, optional): frame rate of the representation model. Defaults to 50.
            segment_duration (float, optional): duration in sec for each audio segment. Defaults to 10.0.
            device (tp.Union[torch.device, str], optional): device. Defaults to 'cpu'.
        """
        from demucs import pretrained
        self.sample_rate = sample_rate
        self.__dict__['demucs'] = pretrained.get_model('htdemucs').to(device)
        stem_sources: list = self.demucs.sources  # type: ignore
        self.stem_idx = stem_sources.index('drums')
        self.compression_model = None
        self.latent_dim = compression_model_latent_dim
        super().__init__(dim=self.latent_dim, output_dim=out_dim, device=device)
        self.autocast = TorchAutocast(enabled=device != 'cpu', device_type=self.device, dtype=torch.float32)
        self._use_masking = False
        self.blurring_factor = blurring_factor
        self.seq_len = int(segment_duration * compression_model_framerate)
        self.cache = None
        if cache_path is not None:
            self.cache = EmbeddingCache(Path(cache_path) / 'wav', self.device,
                                        compute_embed_fn=self._calc_coarse_drum_codes_for_cache,
                                        extract_embed_fn=self._load_drum_codes_chunk)

    @torch.no_grad()
    def _get_drums_stem(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Get parts of the wav that holds the drums, extracting the main stems from the wav."""
        from demucs.apply import apply_model
        from demucs.audio import convert_audio
        with self.autocast:
            wav = convert_audio(
                wav, sample_rate, self.demucs.samplerate, self.demucs.audio_channels)  # type: ignore
            stems = apply_model(self.demucs, wav, device=self.device)
            drum_stem = stems[:, self.stem_idx]  # extract relevant stems for drums conditioning
            return convert_audio(drum_stem, self.demucs.samplerate, self.sample_rate, 1)  # type: ignore

    def _temporal_blur(self, z: torch.Tensor):
        # z: (B, T, C)
        B, T, C = z.shape
        if T % self.blurring_factor != 0:
            # pad with reflect for T % self.temporal_blurring on the right in dim=1
            pad_val = self.blurring_factor - T % self.blurring_factor
            z = torch.nn.functional.pad(z, (0, 0, 0, pad_val), mode='reflect')
        z = z.reshape(B, -1, self.blurring_factor, C).sum(dim=2) / self.blurring_factor
        z = z.unsqueeze(2).repeat(1, 1, self.blurring_factor, 1).reshape(B, -1, C)
        z = z[:, :T]
        assert z.shape == (B, T, C)
        return z

    @torch.no_grad()
    def _extract_coarse_drum_codes(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        assert self.compression_model is not None

        # stem separation of drums
        drums = self._get_drums_stem(wav, sample_rate)

        # continuous encoding with compression model
        latents = self.compression_model.model.encoder(drums)

        # quantization to coarsest codebook
        coarsest_quantizer = self.compression_model.model.quantizer.layers[0]
        drums = coarsest_quantizer.encode(latents).to(torch.int16)
        return drums

    @torch.no_grad()
    def _calc_coarse_drum_codes_for_cache(self, path: tp.Union[str, Path],
                                          x: WavCondition, idx: int,
                                          max_duration_to_process: float = 600) -> torch.Tensor:
        """Extract blurred drum latents from the whole audio waveform at the given path."""
        wav, sr = audio_read(path)
        wav = wav[None].to(self.device)
        wav = convert_audio(wav, sr, self.sample_rate, to_channels=1)

        max_frames_to_process = int(max_duration_to_process * self.sample_rate)
        if wav.shape[-1] > max_frames_to_process:
            # process very long tracks in chunks
            start = 0
            codes = []
            while start < wav.shape[-1] - 1:
                wav_chunk = wav[..., start: start + max_frames_to_process]
                codes.append(self._extract_coarse_drum_codes(wav_chunk, self.sample_rate)[0])
                start += max_frames_to_process
            return torch.cat(codes)

        return self._extract_coarse_drum_codes(wav, self.sample_rate)[0]

    def _load_drum_codes_chunk(self, full_coarse_drum_codes: torch.Tensor, x: WavCondition, idx: int) -> torch.Tensor:
        """Extract a chunk of coarse drum codes from the full coarse drum codes derived from the full waveform."""
        wav_length = x.wav.shape[-1]
        seek_time = x.seek_time[idx]
        assert seek_time is not None, (
            "WavCondition seek_time is required "
            "when extracting chunks from pre-computed drum codes.")
        assert self.compression_model is not None
        frame_rate = self.compression_model.frame_rate
        target_length = int(frame_rate * wav_length / self.sample_rate)
        target_length = max(target_length, self.seq_len)
        index = int(frame_rate * seek_time)
        out = full_coarse_drum_codes[index: index + target_length]
        # pad
        out = torch.cat((out, torch.zeros(target_length - out.shape[0], dtype=out.dtype, device=out.device)))
        return out.to(self.device)

    @torch.no_grad()
    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        bs = x.wav.shape[0]
        if x.wav.shape[-1] <= 1:
            # null condition
            return torch.zeros((bs, self.seq_len, self.latent_dim), device=x.wav.device, dtype=x.wav.dtype)

        # extract coarse drum codes
        no_undefined_paths = all(p is not None for p in x.path)
        no_nullified_cond = x.wav.shape[-1] > 1
        if self.cache is not None and no_undefined_paths and no_nullified_cond:
            paths = [Path(p) for p in x.path if p is not None]
            codes = self.cache.get_embed_from_cache(paths, x)
        else:
            assert all(sr == x.sample_rate[0] for sr in x.sample_rate), "All sample rates in batch should be equal."
            codes = self._extract_coarse_drum_codes(x.wav, x.sample_rate[0])

        assert self.compression_model is not None
        # decode back to the continuous representation of compression model
        codes = codes.unsqueeze(1).permute(1, 0, 2)  # (B, T) -> (1, B, T)
        codes = codes.to(torch.int64)
        latents = self.compression_model.model.quantizer.decode(codes)

        latents = latents.permute(0, 2, 1)  # [B, C, T] -> [B, T, C]

        # temporal blurring
        return self._temporal_blur(latents)

    def tokenize(self, x: WavCondition) -> WavCondition:
        """Apply WavConditioner tokenization and populate cache if needed."""
        x = super().tokenize(x)
        no_undefined_paths = all(p is not None for p in x.path)
        if self.cache is not None and no_undefined_paths:
            paths = [Path(p) for p in x.path if p is not None]
            self.cache.populate_embed_cache(paths, x)
        return x


class JascoConditioningProvider(ConditioningProvider):
    """
    A cond-provider that manages and tokenizes various types of conditioning attributes for Jasco models.
    Attributes:
        chords_card (int): The cardinality of the chord vocabulary.
        sequence_length (int): The length of the sequence for padding purposes.
        melody_dim (int): The dimensionality of the melody matrix.
    """
    def __init__(self, *args,
                 chords_card: int = 194,
                 sequence_length: int = 500,
                 melody_dim: int = 53, **kwargs):
        self.null_chord = chords_card
        self.sequence_len = sequence_length
        self.melody_dim = melody_dim
        super().__init__(*args, **kwargs)

    def tokenize(self, inputs: tp.List[ConditioningAttributes]) -> tp.Dict[str, tp.Any]:
        """Match attributes/wavs with existing conditioners in self, and compute tokenize them accordingly.
        This should be called before starting any real GPU work to avoid synchronization points.
        This will return a dict matching conditioner names to their arbitrary tokenized representations.

        Args:
            inputs (list[ConditioningAttributes]): List of ConditioningAttributes objects containing
                text and wav conditions.
        """
        assert all([isinstance(x, ConditioningAttributes) for x in inputs]), (
            "Got unexpected types input for conditioner! should be tp.List[ConditioningAttributes]",
            f" but types were {set([type(x) for x in inputs])}"
        )

        output = {}
        text = self._collate_text(inputs)
        wavs = self._collate_wavs(inputs)

        symbolic = self._collate_symbolic(inputs, self.conditioners.keys())

        assert set(text.keys() | wavs.keys() | symbolic.keys()).issubset(set(self.conditioners.keys())), (
            f"Got an unexpected attribute! Expected {self.conditioners.keys()}, ",
            f"got {text.keys(), wavs.keys(), symbolic.keys()}"
        )

        for attribute, batch in chain(text.items(), wavs.items(), symbolic.items()):
            output[attribute] = self.conditioners[attribute].tokenize(batch)
        return output

    def _collate_symbolic(self, samples: tp.List[ConditioningAttributes],
                          conditioner_keys: tp.Set) -> tp.Dict[str, SymbolicCondition]:
        output = {}

        # collate if symbolic cond exists
        if any(x in conditioner_keys for x in JascoCondConst.SYM.value):

            for s in samples:
                # hydrate with null chord if chords not exist - for inference support
                if (s.symbolic == {} or
                        s.symbolic[JascoCondConst.CRD.value].frame_chords is None or
                        s.symbolic[JascoCondConst.CRD.value].frame_chords.shape[-1] <= 1):  # type: ignore
                    # no chords conditioning - fill with null chord token
                    s.symbolic[JascoCondConst.CRD.value] = SymbolicCondition(
                        frame_chords=torch.ones(self.sequence_len, dtype=torch.int32) * self.null_chord)

                if (s.symbolic == {} or
                        s.symbolic[JascoCondConst.MLD.value].melody is None or
                        s.symbolic[JascoCondConst.MLD.value].melody.shape[-1] <= 1):  # type: ignore
                    # no chords conditioning - fill with null chord token
                    s.symbolic[JascoCondConst.MLD.value] = SymbolicCondition(
                        melody=torch.zeros((self.melody_dim, self.sequence_len)))

            if JascoCondConst.CRD.value in conditioner_keys:
                # pad to max
                max_seq_len = max(
                    [s.symbolic[JascoCondConst.CRD.value].frame_chords.shape[-1] for s in samples])  # type: ignore
                padded_chords = [
                    torch.cat((x.symbolic[JascoCondConst.CRD.value].frame_chords,   # type: ignore
                               torch.ones(max_seq_len -
                                          x.symbolic[JascoCondConst.CRD.value].frame_chords.shape[-1],  # type: ignore
                                          dtype=torch.int32) * self.null_chord))
                    for x in samples
                ]
                output[JascoCondConst.CRD.value] = SymbolicCondition(frame_chords=torch.stack(padded_chords))
            if JascoCondConst.MLD.value in conditioner_keys:
                melodies = torch.stack([x.symbolic[JascoCondConst.MLD.value].melody for x in samples])  # type: ignore
                output[JascoCondConst.MLD.value] = SymbolicCondition(melody=melodies)
        return output
