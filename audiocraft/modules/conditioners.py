# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
import logging
import math
from pathlib import Path
import random
import re
import typing as tp
import warnings

import einops
from num2words import num2words
import spacy
from transformers import RobertaTokenizer, T5EncoderModel, T5Tokenizer  # type: ignore
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from .chroma import ChromaExtractor
from .streaming import StreamingModule
from .transformer import create_sin_embedding
from ..data.audio import audio_read
from ..data.audio_dataset import SegmentInfo
from ..data.audio_utils import convert_audio
from ..environment import AudioCraftEnvironment
from ..quantization import ResidualVectorQuantizer
from ..utils.autocast import TorchAutocast
from ..utils.cache import EmbeddingCache
from ..utils.utils import collate, hash_trick, length_to_mask, load_clap_state_dict, warn_once


logger = logging.getLogger(__name__)
TextCondition = tp.Optional[str]  # a text condition can be a string or None (if doesn't exist)
ConditionType = tp.Tuple[torch.Tensor, torch.Tensor]  # condition, mask


class WavCondition(tp.NamedTuple):
    wav: torch.Tensor
    length: torch.Tensor
    sample_rate: tp.List[int]
    path: tp.List[tp.Optional[str]] = []
    seek_time: tp.List[tp.Optional[float]] = []


class JointEmbedCondition(tp.NamedTuple):
    wav: torch.Tensor
    text: tp.List[tp.Optional[str]]
    length: torch.Tensor
    sample_rate: tp.List[int]
    path: tp.List[tp.Optional[str]] = []
    seek_time: tp.List[tp.Optional[float]] = []


@dataclass
class ConditioningAttributes:
    text: tp.Dict[str, tp.Optional[str]] = field(default_factory=dict)
    wav: tp.Dict[str, WavCondition] = field(default_factory=dict)
    joint_embed: tp.Dict[str, JointEmbedCondition] = field(default_factory=dict)

    def __getitem__(self, item):
        return getattr(self, item)

    @property
    def text_attributes(self):
        return self.text.keys()

    @property
    def wav_attributes(self):
        return self.wav.keys()

    @property
    def joint_embed_attributes(self):
        return self.joint_embed.keys()

    @property
    def attributes(self):
        return {
            "text": self.text_attributes,
            "wav": self.wav_attributes,
            "joint_embed": self.joint_embed_attributes,
        }

    def to_flat_dict(self):
        return {
            **{f"text.{k}": v for k, v in self.text.items()},
            **{f"wav.{k}": v for k, v in self.wav.items()},
            **{f"joint_embed.{k}": v for k, v in self.joint_embed.items()}
        }

    @classmethod
    def from_flat_dict(cls, x):
        out = cls()
        for k, v in x.items():
            kind, att = k.split(".")
            out[kind][att] = v
        return out


class SegmentWithAttributes(SegmentInfo):
    """Base class for all dataclasses that are used for conditioning.
    All child classes should implement `to_condition_attributes` that converts
    the existing attributes to a dataclass of type ConditioningAttributes.
    """
    def to_condition_attributes(self) -> ConditioningAttributes:
        raise NotImplementedError()


def nullify_condition(condition: ConditionType, dim: int = 1):
    """Transform an input condition to a null condition.
    The way it is done by converting it to a single zero vector similarly
    to how it is done inside WhiteSpaceTokenizer and NoopTokenizer.

    Args:
        condition (ConditionType): A tuple of condition and mask (tuple[torch.Tensor, torch.Tensor])
        dim (int): The dimension that will be truncated (should be the time dimension)
        WARNING!: dim should not be the batch dimension!
    Returns:
        ConditionType: A tuple of null condition and mask
    """
    assert dim != 0, "dim cannot be the batch dimension!"
    assert isinstance(condition, tuple) and \
        isinstance(condition[0], torch.Tensor) and \
        isinstance(condition[1], torch.Tensor), "'nullify_condition' got an unexpected input type!"
    cond, mask = condition
    B = cond.shape[0]
    last_dim = cond.dim() - 1
    out = cond.transpose(dim, last_dim)
    out = 0. * out[..., :1]
    out = out.transpose(dim, last_dim)
    mask = torch.zeros((B, 1), device=out.device).int()
    assert cond.dim() == out.dim()
    return out, mask


def nullify_wav(cond: WavCondition) -> WavCondition:
    """Transform a WavCondition to a nullified WavCondition.
    It replaces the wav by a null tensor, forces its length to 0, and replaces metadata by dummy attributes.

    Args:
        cond (WavCondition): Wav condition with wav, tensor of shape [B, T].
    Returns:
        WavCondition: Nullified wav condition.
    """
    null_wav, _ = nullify_condition((cond.wav, torch.zeros_like(cond.wav)), dim=cond.wav.dim() - 1)
    return WavCondition(
        wav=null_wav,
        length=torch.tensor([0] * cond.wav.shape[0], device=cond.wav.device),
        sample_rate=cond.sample_rate,
        path=[None] * cond.wav.shape[0],
        seek_time=[None] * cond.wav.shape[0],
    )


def nullify_joint_embed(embed: JointEmbedCondition) -> JointEmbedCondition:
    """Nullify the joint embedding condition by replacing it by a null tensor, forcing its length to 0,
    and replacing metadata by dummy attributes.

    Args:
        cond (JointEmbedCondition): Joint embedding condition with wav and text, wav tensor of shape [B, C, T].
    """
    null_wav, _ = nullify_condition((embed.wav, torch.zeros_like(embed.wav)), dim=embed.wav.dim() - 1)
    return JointEmbedCondition(
        wav=null_wav, text=[None] * len(embed.text),
        length=torch.LongTensor([0]).to(embed.wav.device),
        sample_rate=embed.sample_rate,
        path=[None] * embed.wav.shape[0],
        seek_time=[0] * embed.wav.shape[0],
    )


class Tokenizer:
    """Base tokenizer implementation
    (in case we want to introduce more advances tokenizers in the future).
    """
    def __call__(self, texts: tp.List[tp.Optional[str]]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class WhiteSpaceTokenizer(Tokenizer):
    """This tokenizer should be used for natural language descriptions.
    For example:
    ["he didn't, know he's going home.", 'shorter sentence'] =>
    [[78, 62, 31,  4, 78, 25, 19, 34],
    [59, 77,  0,  0,  0,  0,  0,  0]]
    """
    PUNCTUATION = "?:!.,;"

    def __init__(self, n_bins: int, pad_idx: int = 0, language: str = "en_core_web_sm",
                 lemma: bool = True, stopwords: bool = True) -> None:
        self.n_bins = n_bins
        self.pad_idx = pad_idx
        self.lemma = lemma
        self.stopwords = stopwords
        try:
            self.nlp = spacy.load(language)
        except IOError:
            spacy.cli.download(language)  # type: ignore
            self.nlp = spacy.load(language)

    @tp.no_type_check
    def __call__(self, texts: tp.List[tp.Optional[str]],
                 return_text: bool = False) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Take a list of strings and convert them to a tensor of indices.

        Args:
            texts (list[str]): List of strings.
            return_text (bool, optional): Whether to return text as additional tuple item. Defaults to False.
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Indices of words in the LUT.
                - And a mask indicating where the padding tokens are
        """
        output, lengths = [], []
        texts = deepcopy(texts)
        for i, text in enumerate(texts):
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(torch.Tensor([self.pad_idx]))
                lengths.append(0)
                continue

            # convert numbers to words
            text = re.sub(r"(\d+)", lambda x: num2words(int(x.group(0))), text)  # type: ignore
            # normalize text
            text = self.nlp(text)  # type: ignore
            # remove stopwords
            if self.stopwords:
                text = [w for w in text if not w.is_stop]  # type: ignore
            # remove punctuation
            text = [w for w in text if w.text not in self.PUNCTUATION]  # type: ignore
            # lemmatize if needed
            text = [getattr(t, "lemma_" if self.lemma else "text") for t in text]  # type: ignore

            texts[i] = " ".join(text)
            lengths.append(len(text))
            # convert to tensor
            tokens = torch.Tensor([hash_trick(w, self.n_bins) for w in text])
            output.append(tokens)

        mask = length_to_mask(torch.IntTensor(lengths)).int()
        padded_output = pad_sequence(output, padding_value=self.pad_idx).int().t()
        if return_text:
            return padded_output, mask, texts  # type: ignore
        return padded_output, mask


class NoopTokenizer(Tokenizer):
    """This tokenizer should be used for global conditioners such as: artist, genre, key, etc.
    The difference between this and WhiteSpaceTokenizer is that NoopTokenizer does not split
    strings, so "Jeff Buckley" will get it's own index. Whereas WhiteSpaceTokenizer will
    split it to ["Jeff", "Buckley"] and return an index per word.

    For example:
    ["Queen", "ABBA", "Jeff Buckley"] => [43, 55, 101]
    ["Metal", "Rock", "Classical"] => [0, 223, 51]
    """
    def __init__(self, n_bins: int, pad_idx: int = 0):
        self.n_bins = n_bins
        self.pad_idx = pad_idx

    def __call__(self, texts: tp.List[tp.Optional[str]]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        output, lengths = [], []
        for text in texts:
            # if current sample doesn't have a certain attribute, replace with pad token
            if text is None:
                output.append(self.pad_idx)
                lengths.append(0)
            else:
                output.append(hash_trick(text, self.n_bins))
                lengths.append(1)

        tokens = torch.LongTensor(output).unsqueeze(1)
        mask = length_to_mask(torch.IntTensor(lengths)).int()
        return tokens, mask


class BaseConditioner(nn.Module):
    """Base model for all conditioner modules.
    We allow the output dim to be different than the hidden dim for two reasons:
    1) keep our LUTs small when the vocab is large;
    2) make all condition dims consistent.

    Args:
        dim (int): Hidden dim of the model.
        output_dim (int): Output dim of the conditioner.
    """
    def __init__(self, dim: int, output_dim: int):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.output_proj = nn.Linear(dim, output_dim)

    def tokenize(self, *args, **kwargs) -> tp.Any:
        """Should be any part of the processing that will lead to a synchronization
        point, e.g. BPE tokenization with transfer to the GPU.

        The returned value will be saved and return later when calling forward().
        """
        raise NotImplementedError()

    def forward(self, inputs: tp.Any) -> ConditionType:
        """Gets input that should be used as conditioning (e.g, genre, description or a waveform).
        Outputs a ConditionType, after the input data was embedded as a dense vector.

        Returns:
            ConditionType:
                - A tensor of size [B, T, D] where B is the batch size, T is the length of the
                  output embedding and D is the dimension of the embedding.
                - And a mask indicating where the padding tokens.
        """
        raise NotImplementedError()


class TextConditioner(BaseConditioner):
    ...


class LUTConditioner(TextConditioner):
    """Lookup table TextConditioner.

    Args:
        n_bins (int): Number of bins.
        dim (int): Hidden dim of the model (text-encoder/LUT).
        output_dim (int): Output dim of the conditioner.
        tokenizer (str): Name of the tokenizer.
        pad_idx (int, optional): Index for padding token. Defaults to 0.
    """
    def __init__(self, n_bins: int, dim: int, output_dim: int, tokenizer: str, pad_idx: int = 0):
        super().__init__(dim, output_dim)
        self.embed = nn.Embedding(n_bins, dim)
        self.tokenizer: Tokenizer
        if tokenizer == 'whitespace':
            self.tokenizer = WhiteSpaceTokenizer(n_bins, pad_idx=pad_idx)
        elif tokenizer == 'noop':
            self.tokenizer = NoopTokenizer(n_bins, pad_idx=pad_idx)
        else:
            raise ValueError(f"unrecognized tokenizer `{tokenizer}`.")

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        device = self.embed.weight.device
        tokens, mask = self.tokenizer(x)
        tokens, mask = tokens.to(device), mask.to(device)
        return tokens, mask

    def forward(self, inputs: tp.Tuple[torch.Tensor, torch.Tensor]) -> ConditionType:
        tokens, mask = inputs
        embeds = self.embed(tokens)
        embeds = self.output_proj(embeds)
        embeds = (embeds * mask.unsqueeze(-1))
        return embeds, mask


class T5Conditioner(TextConditioner):
    """T5-based TextConditioner.

    Args:
        name (str): Name of the T5 model.
        output_dim (int): Output dim of the conditioner.
        finetune (bool): Whether to fine-tune T5 at train time.
        device (str): Device for T5 Conditioner.
        autocast_dtype (tp.Optional[str], optional): Autocast dtype.
        word_dropout (float, optional): Word dropout probability.
        normalize_text (bool, optional): Whether to apply text normalization.
    """
    MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"]
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }

    def __init__(self, name: str, output_dim: int, finetune: bool, device: str,
                 autocast_dtype: tp.Optional[str] = 'float32', word_dropout: float = 0.,
                 normalize_text: bool = False):
        assert name in self.MODELS, f"Unrecognized t5 model name (should in {self.MODELS})"
        super().__init__(self.MODELS_DIMS[name], output_dim)
        self.device = device
        self.name = name
        self.finetune = finetune
        self.word_dropout = word_dropout
        if autocast_dtype is None or self.device == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
            if self.device != 'cpu':
                logger.warning("T5 has no autocast, this might lead to NaN")
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            logger.info(f"T5 will be evaluated with autocast as {autocast_dtype}")
            self.autocast = TorchAutocast(enabled=True, device_type=self.device, dtype=dtype)
        # Let's disable logging temporarily because T5 will vomit some errors otherwise.
        # thanks https://gist.github.com/simon-weber/7853144
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.t5_tokenizer = T5Tokenizer.from_pretrained(name)
                t5 = T5EncoderModel.from_pretrained(name).train(mode=finetune)
            finally:
                logging.disable(previous_level)
        if finetune:
            self.t5 = t5
        else:
            # this makes sure that the t5 models is not part
            # of the saved checkpoint
            self.__dict__['t5'] = t5.to(device)

        self.normalize_text = normalize_text
        if normalize_text:
            self.text_normalizer = WhiteSpaceTokenizer(1, lemma=True, stopwords=True)

    def tokenize(self, x: tp.List[tp.Optional[str]]) -> tp.Dict[str, torch.Tensor]:
        # if current sample doesn't have a certain attribute, replace with empty string
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]
        if self.normalize_text:
            _, _, entries = self.text_normalizer(entries, return_text=True)
        if self.word_dropout > 0. and self.training:
            new_entries = []
            for entry in entries:
                words = [word for word in entry.split(" ") if random.random() >= self.word_dropout]
                new_entries.append(" ".join(words))
            entries = new_entries

        empty_idx = torch.LongTensor([i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.t5_tokenizer(entries, return_tensors='pt', padding=True).to(self.device)
        mask = inputs['attention_mask']
        mask[empty_idx, :] = 0  # zero-out index where the input is non-existant
        return inputs

    def forward(self, inputs: tp.Dict[str, torch.Tensor]) -> ConditionType:
        mask = inputs['attention_mask']
        with torch.set_grad_enabled(self.finetune), self.autocast:
            embeds = self.t5(**inputs).last_hidden_state
        embeds = self.output_proj(embeds.to(self.output_proj.weight))
        embeds = (embeds * mask.unsqueeze(-1))
        return embeds, mask


class WaveformConditioner(BaseConditioner):
    """Base class for all conditioners that take a waveform as input.
    Classes that inherit must implement `_get_wav_embedding` that outputs
    a continuous tensor, and `_downsampling_factor` that returns the down-sampling
    factor of the embedding model.

    Args:
        dim (int): The internal representation dimension.
        output_dim (int): Output dimension.
        device (tp.Union[torch.device, str]): Device.
    """
    def __init__(self, dim: int, output_dim: int, device: tp.Union[torch.device, str]):
        super().__init__(dim, output_dim)
        self.device = device

    def tokenize(self, x: WavCondition) -> WavCondition:
        wav, length, sample_rate, path, seek_time = x
        assert length is not None
        return WavCondition(wav.to(self.device), length.to(self.device), sample_rate, path, seek_time)

    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        """Gets as input a WavCondition and returns a dense embedding."""
        raise NotImplementedError()

    def _downsampling_factor(self):
        """Returns the downsampling factor of the embedding model."""
        raise NotImplementedError()

    def forward(self, x: WavCondition) -> ConditionType:
        """Extract condition embedding and mask from a waveform and its metadata.
        Args:
            x (WavCondition): Waveform condition containing raw waveform and metadata.
        Returns:
            ConditionType: a dense vector representing the conditioning along with its mask
        """
        wav, lengths, *_ = x
        with torch.no_grad():
            embeds = self._get_wav_embedding(x)
        embeds = embeds.to(self.output_proj.weight)
        embeds = self.output_proj(embeds)

        if lengths is not None:
            lengths = lengths / self._downsampling_factor()
            mask = length_to_mask(lengths, max_len=embeds.shape[1]).int()  # type: ignore
        else:
            mask = torch.ones_like(embeds)
        embeds = (embeds * mask.unsqueeze(2).to(self.device))

        return embeds, mask


class ChromaStemConditioner(WaveformConditioner):
    """Chroma conditioner based on stems.
    The ChromaStemConditioner uses DEMUCS to first filter out drums and bass, as
    the drums and bass often dominate the chroma leading to the chroma features
    not containing information about the melody.

    Args:
        output_dim (int): Output dimension for the conditioner.
        sample_rate (int): Sample rate for the chroma extractor.
        n_chroma (int): Number of chroma bins for the chroma extractor.
        radix2_exp (int): Size of stft window for the chroma extractor (power of 2, e.g. 12 -> 2^12).
        duration (int): duration used during training. This is later used for correct padding
            in case we are using chroma as prefix.
        match_len_on_eval (bool, optional): if True then all chromas are padded to the training
            duration. Defaults to False.
        eval_wavs (str, optional): path to a dataset manifest with waveform, this waveforms are used as
            conditions during eval (for cases where we don't want to leak test conditions like MusicCaps).
            Defaults to None.
        n_eval_wavs (int, optional): limits the number of waveforms used for conditioning. Defaults to 0.
        device (tp.Union[torch.device, str], optional): Device for the conditioner.
        **kwargs: Additional parameters for the chroma extractor.
    """
    def __init__(self, output_dim: int, sample_rate: int, n_chroma: int, radix2_exp: int,
                 duration: float, match_len_on_eval: bool = True, eval_wavs: tp.Optional[str] = None,
                 n_eval_wavs: int = 0, cache_path: tp.Optional[tp.Union[str, Path]] = None,
                 device: tp.Union[torch.device, str] = 'cpu', **kwargs):
        from demucs import pretrained
        super().__init__(dim=n_chroma, output_dim=output_dim, device=device)
        self.autocast = TorchAutocast(enabled=device != 'cpu', device_type=self.device, dtype=torch.float32)
        self.sample_rate = sample_rate
        self.match_len_on_eval = match_len_on_eval
        self.duration = duration
        self.__dict__['demucs'] = pretrained.get_model('htdemucs').to(device)
        stem_sources: list = self.demucs.sources  # type: ignore
        self.stem_indices = torch.LongTensor([stem_sources.index('vocals'), stem_sources.index('other')]).to(device)
        self.chroma = ChromaExtractor(sample_rate=sample_rate, n_chroma=n_chroma,
                                      radix2_exp=radix2_exp, **kwargs).to(device)
        self.chroma_len = self._get_chroma_len()
        self.eval_wavs: tp.Optional[torch.Tensor] = self._load_eval_wavs(eval_wavs, n_eval_wavs)
        self.cache = None
        if cache_path is not None:
            self.cache = EmbeddingCache(Path(cache_path) / 'wav', self.device,
                                        compute_embed_fn=self._get_full_chroma_for_cache,
                                        extract_embed_fn=self._extract_chroma_chunk)

    def _downsampling_factor(self) -> int:
        return self.chroma.winhop

    def _load_eval_wavs(self, path: tp.Optional[str], num_samples: int) -> tp.Optional[torch.Tensor]:
        """Load pre-defined waveforms from a json.
        These waveforms will be used for chroma extraction during evaluation.
        This is done to make the evaluation on MusicCaps fair (we shouldn't see the chromas of MusicCaps).
        """
        if path is None:
            return None

        logger.info(f"Loading evaluation wavs from {path}")
        from audiocraft.data.audio_dataset import AudioDataset
        dataset: AudioDataset = AudioDataset.from_meta(
            path, segment_duration=self.duration, min_audio_duration=self.duration,
            sample_rate=self.sample_rate, channels=1)

        if len(dataset) > 0:
            eval_wavs = dataset.collater([dataset[i] for i in range(num_samples)]).to(self.device)
            logger.info(f"Using {len(eval_wavs)} evaluation wavs for chroma-stem conditioner")
            return eval_wavs
        else:
            raise ValueError("Could not find evaluation wavs, check lengths of wavs")

    def reset_eval_wavs(self, eval_wavs: tp.Optional[torch.Tensor]) -> None:
        self.eval_wavs = eval_wavs

    def has_eval_wavs(self) -> bool:
        return self.eval_wavs is not None

    def _sample_eval_wavs(self, num_samples: int) -> torch.Tensor:
        """Sample wavs from a predefined list."""
        assert self.eval_wavs is not None, "Cannot sample eval wavs as no eval wavs provided."
        total_eval_wavs = len(self.eval_wavs)
        out = self.eval_wavs
        if num_samples > total_eval_wavs:
            out = self.eval_wavs.repeat(num_samples // total_eval_wavs + 1, 1, 1)
        return out[torch.randperm(len(out))][:num_samples]

    def _get_chroma_len(self) -> int:
        """Get length of chroma during training."""
        dummy_wav = torch.zeros((1, int(self.sample_rate * self.duration)), device=self.device)
        dummy_chr = self.chroma(dummy_wav)
        return dummy_chr.shape[1]

    @torch.no_grad()
    def _get_stemmed_wav(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Get parts of the wav that holds the melody, extracting the main stems from the wav."""
        from demucs.apply import apply_model
        from demucs.audio import convert_audio
        with self.autocast:
            wav = convert_audio(
                wav, sample_rate, self.demucs.samplerate, self.demucs.audio_channels)  # type: ignore
            stems = apply_model(self.demucs, wav, device=self.device)
            stems = stems[:, self.stem_indices]  # extract relevant stems for melody conditioning
            mix_wav = stems.sum(1)  # merge extracted stems to single waveform
            mix_wav = convert_audio(mix_wav, self.demucs.samplerate, self.sample_rate, 1)  # type: ignore
            return mix_wav

    @torch.no_grad()
    def _extract_chroma(self, wav: torch.Tensor) -> torch.Tensor:
        """Extract chroma features from the waveform."""
        with self.autocast:
            return self.chroma(wav)

    @torch.no_grad()
    def _compute_wav_embedding(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Compute wav embedding, applying stem and chroma extraction."""
        # avoid 0-size tensors when we are working with null conds
        if wav.shape[-1] == 1:
            return self._extract_chroma(wav)
        stems = self._get_stemmed_wav(wav, sample_rate)
        chroma = self._extract_chroma(stems)
        return chroma

    @torch.no_grad()
    def _get_full_chroma_for_cache(self, path: tp.Union[str, Path], x: WavCondition, idx: int) -> torch.Tensor:
        """Extract chroma from the whole audio waveform at the given path."""
        wav, sr = audio_read(path)
        wav = wav[None].to(self.device)
        wav = convert_audio(wav, sr, self.sample_rate, to_channels=1)
        chroma = self._compute_wav_embedding(wav, self.sample_rate)[0]
        return chroma

    def _extract_chroma_chunk(self, full_chroma: torch.Tensor, x: WavCondition, idx: int) -> torch.Tensor:
        """Extract a chunk of chroma from the full chroma derived from the full waveform."""
        wav_length = x.wav.shape[-1]
        seek_time = x.seek_time[idx]
        assert seek_time is not None, (
            "WavCondition seek_time is required "
            "when extracting chroma chunks from pre-computed chroma.")
        full_chroma = full_chroma.float()
        frame_rate = self.sample_rate / self._downsampling_factor()
        target_length = int(frame_rate * wav_length / self.sample_rate)
        index = int(frame_rate * seek_time)
        out = full_chroma[index: index + target_length]
        out = F.pad(out[None], (0, 0, 0, target_length - out.shape[0]))[0]
        return out.to(self.device)

    @torch.no_grad()
    def _get_wav_embedding(self, x: WavCondition) -> torch.Tensor:
        """Get the wav embedding from the WavCondition.
        The conditioner will either extract the embedding on-the-fly computing it from the condition wav directly
        or will rely on the embedding cache to load the pre-computed embedding if relevant.
        """
        sampled_wav: tp.Optional[torch.Tensor] = None
        if not self.training and self.eval_wavs is not None:
            warn_once(logger, "Using precomputed evaluation wavs!")
            sampled_wav = self._sample_eval_wavs(len(x.wav))

        no_undefined_paths = all(p is not None for p in x.path)
        no_nullified_cond = x.wav.shape[-1] > 1
        if sampled_wav is not None:
            chroma = self._compute_wav_embedding(sampled_wav, self.sample_rate)
        elif self.cache is not None and no_undefined_paths and no_nullified_cond:
            paths = [Path(p) for p in x.path if p is not None]
            chroma = self.cache.get_embed_from_cache(paths, x)
        else:
            assert all(sr == x.sample_rate[0] for sr in x.sample_rate), "All sample rates in batch should be equal."
            chroma = self._compute_wav_embedding(x.wav, x.sample_rate[0])

        if self.match_len_on_eval:
            B, T, C = chroma.shape
            if T > self.chroma_len:
                chroma = chroma[:, :self.chroma_len]
                logger.debug(f"Chroma was truncated to match length! ({T} -> {chroma.shape[1]})")
            elif T < self.chroma_len:
                n_repeat = int(math.ceil(self.chroma_len / T))
                chroma = chroma.repeat(1, n_repeat, 1)
                chroma = chroma[:, :self.chroma_len]
                logger.debug(f"Chroma was repeated to match length! ({T} -> {chroma.shape[1]})")

        return chroma

    def tokenize(self, x: WavCondition) -> WavCondition:
        """Apply WavConditioner tokenization and populate cache if needed."""
        x = super().tokenize(x)
        no_undefined_paths = all(p is not None for p in x.path)
        if self.cache is not None and no_undefined_paths:
            paths = [Path(p) for p in x.path if p is not None]
            self.cache.populate_embed_cache(paths, x)
        return x


class JointEmbeddingConditioner(BaseConditioner):
    """Joint embedding conditioning supporting both audio or text conditioning.

    Args:
        dim (int): Dimension.
        output_dim (int): Output dimension.
        device (str): Device.
        attribute (str): Attribute used by the conditioner.
        autocast_dtype (str): Autocast for the conditioner.
        quantize (bool): Whether to quantize the CLAP embedding.
        n_q (int): Number of residual quantizers (used if quantize is true).
        bins (int): Quantizers' codebooks size (used if quantize is true).
        kwargs: Additional parameters for residual vector quantizer.
    """
    def __init__(self, dim: int, output_dim: int, device: str, attribute: str,
                 autocast_dtype: tp.Optional[str] = 'float32', quantize: bool = True,
                 n_q: int = 12, bins: int = 1024, **kwargs):
        super().__init__(dim=dim, output_dim=output_dim)
        self.device = device
        self.attribute = attribute
        if autocast_dtype is None or device == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
            logger.warning("JointEmbeddingConditioner has no autocast, this might lead to NaN.")
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            logger.info(f"JointEmbeddingConditioner will be evaluated with autocast as {autocast_dtype}.")
            self.autocast = TorchAutocast(enabled=True, device_type=self.device, dtype=dtype)
        # residual vector quantizer to discretize the conditioned embedding
        self.quantizer: tp.Optional[ResidualVectorQuantizer] = None
        if quantize:
            self.quantizer = ResidualVectorQuantizer(dim, n_q=n_q, bins=bins, **kwargs)

    def _get_embed(self, x: JointEmbedCondition) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Get joint embedding in latent space from the inputs.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tensor for the latent embedding
                and corresponding empty indexes.
        """
        raise NotImplementedError()

    def forward(self, x: JointEmbedCondition) -> ConditionType:
        with self.autocast:
            embed, empty_idx = self._get_embed(x)
            if self.quantizer is not None:
                embed = embed.view(-1, self.dim, 1)
                q_res = self.quantizer(embed, frame_rate=1)
                out_embed = q_res.x.view(-1, self.dim)
            else:
                out_embed = embed
            out_embed = self.output_proj(out_embed).view(-1, 1, self.output_dim)
            mask = torch.ones(*out_embed.shape[:2], device=out_embed.device)
            mask[empty_idx, :] = 0  # zero-out index where the input is non-existant
            out_embed = (out_embed * mask.unsqueeze(-1))
            return out_embed, mask

    def tokenize(self, x: JointEmbedCondition) -> JointEmbedCondition:
        return x


class CLAPEmbeddingConditioner(JointEmbeddingConditioner):
    """Joint Embedding conditioner based on pre-trained CLAP model.

    This CLAP-based conditioner supports a caching mechanism
    over the computed embeddings for faster training.

    Args:
        dim (int): Dimension.
        output_dim (int): Output dimension.
        device (str): Device.
        attribute (str): Attribute used by the conditioner.
        quantize (bool): Whether to quantize the CLAP embedding.
        n_q (int): Number of residual quantizers (used if quantize is true).
        bins (int): Quantizers' codebooks size (used if quantize is true).
        checkpoint (str): Path to CLAP checkpoint.
        model_arch (str): CLAP model architecture.
        enable_fusion (bool): Enable fusion for CLAP model.
        sample_rate (int): Sample rate used by CLAP model.
        max_audio_length (float): Maximum audio length for CLAP model.
        audio_stride (float): Stride to use for getting a CLAP embedding on the full sequence.
        normalize (bool): Whether to normalize the CLAP embedding.
        text_p (float): Probability of using text representation instead of audio at train time.
        batch_size (Optional[int]): Batch size for CLAP embedding computation.
        autocast_dtype (str): Autocast for the conditioner.
        cache_path (Optional[str]): Path for pre-computed embeddings caching.
        kwargs: Additional parameters for residual vector quantizer.
    """
    def __init__(self, dim: int, output_dim: int, device: str, attribute: str,
                 quantize: bool, n_q: int, bins: int, checkpoint: tp.Union[str, Path], model_arch: str,
                 enable_fusion: bool, sample_rate: int, max_audio_length: int, audio_stride: int,
                 normalize: bool, text_p: bool, batch_size: tp.Optional[int] = None,
                 autocast_dtype: tp.Optional[str] = 'float32', cache_path: tp.Optional[str] = None, **kwargs):
        try:
            import laion_clap  # type: ignore
        except ImportError:
            raise ImportError("Please install CLAP to use the CLAPEmbeddingConditioner: 'pip install laion_clap'")
        warnings.warn("Sample rate for CLAP conditioner was fixed in version v1.1.0, (from 44.1 to 48 kHz). "
                      "Please retrain all models.")
        checkpoint = AudioCraftEnvironment.resolve_reference_path(checkpoint)
        clap_tokenize = RobertaTokenizer.from_pretrained('roberta-base')
        clap_model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=model_arch)
        load_clap_state_dict(clap_model, checkpoint)
        clap_model.eval()
        clap_model.to(device)
        super().__init__(dim=dim, output_dim=output_dim, device=device, attribute=attribute,
                         autocast_dtype=autocast_dtype, quantize=quantize, n_q=n_q, bins=bins,
                         **kwargs)
        self.checkpoint = checkpoint
        self.enable_fusion = enable_fusion
        self.model_arch = model_arch
        self.clap: laion_clap.CLAP_Module
        self.clap_tokenize: RobertaTokenizer
        self.clap_sample_rate = sample_rate
        self.clap_max_frames = int(self.clap_sample_rate * max_audio_length)
        self.clap_stride = int(self.clap_sample_rate * audio_stride)
        self.batch_size = batch_size or 1
        self.normalize = normalize
        self.text_p = text_p
        self.__dict__['clap_tokenize'] = clap_tokenize
        self.__dict__['clap'] = clap_model
        self.wav_cache, self.text_cache = None, None
        if cache_path is not None:
            self.wav_cache = EmbeddingCache(Path(cache_path) / 'wav', self.device,
                                            compute_embed_fn=self._get_wav_embedding_for_cache,
                                            extract_embed_fn=self._extract_wav_embedding_chunk)
            self.text_cache = EmbeddingCache(Path(cache_path) / 'text', self.device,
                                             compute_embed_fn=self._get_text_embedding_for_cache)

    def _tokenizer(self, texts: tp.Union[str, tp.List[str]]) -> dict:
        # we use the default params from CLAP module here as well
        return self.clap_tokenize(texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")

    def _compute_text_embedding(self, text: tp.List[str]) -> torch.Tensor:
        """Compute text embedding from CLAP model on a given a batch of text.

        Args:
            text (list[str]): List of text for the batch, with B items.
        Returns:
            torch.Tensor: CLAP embedding derived from text, of shape [B, 1, D], with D the CLAP embedding dimension.
        """
        with torch.no_grad():
            embed = self.clap.get_text_embedding(text, tokenizer=self._tokenizer, use_tensor=True)
            return embed.view(embed.size(0), 1, embed.size(-1))

    def _get_text_embedding_for_cache(self, path: tp.Union[Path, str],
                                      x: JointEmbedCondition, idx: int) -> torch.Tensor:
        """Get text embedding function for the cache."""
        text = x.text[idx]
        text = text if text is not None else ""
        return self._compute_text_embedding([text])[0]

    def _preprocess_wav(self, wav: torch.Tensor, length: torch.Tensor, sample_rates: tp.List[int]) -> torch.Tensor:
        """Preprocess wav to expected format by CLAP model.

        Args:
            wav (torch.Tensor): Audio wav, of shape [B, C, T].
            length (torch.Tensor): Actual length of the audio for each item in the batch, of shape [B].
            sample_rates (list[int]): Sample rates for each sample in the batch
        Returns:
            torch.Tensor: Audio wav of shape [B, T].
        """
        assert wav.dim() == 3, "Expecting wav to be [B, C, T]"
        if sample_rates is not None:
            _wav = []
            for i, audio in enumerate(wav):
                sr = sample_rates[i]
                audio = convert_audio(audio, from_rate=sr, to_rate=self.clap_sample_rate, to_channels=1)
                _wav.append(audio)
            wav = torch.stack(_wav, dim=0)
        wav = wav.mean(dim=1)
        return wav

    def _compute_wav_embedding(self, wav: torch.Tensor, length: torch.Tensor,
                               sample_rates: tp.List[int], reduce_mean: bool = False) -> torch.Tensor:
        """Compute audio wave embedding from CLAP model.

        Since CLAP operates on a fixed sequence length audio inputs and we need to process longer audio sequences,
        we calculate the wav embeddings on `clap_max_frames` windows with `clap_stride`-second stride and
        average the resulting embeddings.

        Args:
            wav (torch.Tensor): Audio wav, of shape [B, C, T].
            length (torch.Tensor): Actual length of the audio for each item in the batch, of shape [B].
            sample_rates (list[int]): Sample rates for each sample in the batch.
            reduce_mean (bool): Whether to get the average tensor.
        Returns:
            torch.Tensor: Audio embedding of shape [B, F, D], F being the number of chunks, D the dimension.
        """
        with torch.no_grad():
            wav = self._preprocess_wav(wav, length, sample_rates)
            B, T = wav.shape
            if T >= self.clap_max_frames:
                wav = wav.unfold(-1, self.clap_max_frames, self.clap_stride)  # [B, F, T]
            else:
                wav = wav.view(-1, 1, T)  # [B, F, T] with F=1
            wav = einops.rearrange(wav, 'b f t -> (b f) t')
            embed_list = []
            for i in range(0, wav.size(0), self.batch_size):
                _wav = wav[i:i+self.batch_size, ...]
                _embed = self.clap.get_audio_embedding_from_data(_wav, use_tensor=True)
                embed_list.append(_embed)
            embed = torch.cat(embed_list, dim=0)
            embed = einops.rearrange(embed, '(b f) d -> b f d', b=B)
            if reduce_mean:
                embed = embed.mean(dim=1, keepdim=True)
            return embed  # [B, F, D] with F=1 if reduce_mean is True

    def _get_wav_embedding_for_cache(self, path: tp.Union[str, Path],
                                     x: JointEmbedCondition, idx: int) -> torch.Tensor:
        """Compute audio wave embedding for the cache.
        The embedding is computed on a given audio read from file.

        Args:
            path (str or Path): Path to the full audio file.
        Returns:
            torch.Tensor: Single-item tensor of shape [F, D], F being the number of chunks, D the dimension.
        """
        wav, sr = audio_read(path)  # [C, T]
        wav = wav.unsqueeze(0).to(self.device)  # [1, C, T]
        wav_len = torch.LongTensor([wav.shape[-1]]).to(self.device)
        embed = self._compute_wav_embedding(wav, wav_len, [sr], reduce_mean=False)  # [B, F, D]
        return embed.squeeze(0)  # [F, D]

    def _extract_wav_embedding_chunk(self, full_embed: torch.Tensor, x: JointEmbedCondition, idx: int) -> torch.Tensor:
        """Extract the chunk of embedding matching the seek_time and length from the full CLAP audio embedding.

        Args:
            full_embed (torch.Tensor): CLAP embedding computed on the full wave, of shape [F, D].
            x (JointEmbedCondition): Joint embedding condition for the full batch.
            idx (int): Index considered for the given embedding to extract.
        Returns:
            torch.Tensor: Wav embedding averaged on sliding window, of shape [1, D].
        """
        sample_rate = x.sample_rate[idx]
        seek_time = x.seek_time[idx]
        seek_time = 0. if seek_time is None else seek_time
        clap_stride = int(self.clap_stride / self.clap_sample_rate) * sample_rate
        end_seek_time = seek_time + self.clap_max_frames / self.clap_sample_rate
        start_offset = int(seek_time * sample_rate // clap_stride)
        end_offset = int(end_seek_time * sample_rate // clap_stride)
        wav_embed = full_embed[start_offset:end_offset, ...]
        wav_embed = wav_embed.mean(dim=0, keepdim=True)
        return wav_embed.to(self.device)  # [F, D]

    def _get_text_embedding(self, x: JointEmbedCondition) -> torch.Tensor:
        """Get CLAP embedding from a batch of text descriptions."""
        no_nullified_cond = x.wav.shape[-1] > 1  # we don't want to read from cache when condition dropout
        if self.text_cache is not None and no_nullified_cond:
            assert all(p is not None for p in x.path), "Cache requires all JointEmbedCondition paths to be provided"
            paths = [Path(p) for p in x.path if p is not None]
            embed = self.text_cache.get_embed_from_cache(paths, x)
        else:
            text = [xi if xi is not None else "" for xi in x.text]
            embed = self._compute_text_embedding(text)
        if self.normalize:
            embed = torch.nn.functional.normalize(embed, p=2.0, dim=-1)
        return embed

    def _get_wav_embedding(self, x: JointEmbedCondition) -> torch.Tensor:
        """Get CLAP embedding from a batch of audio tensors (and corresponding sample rates)."""
        no_undefined_paths = all(p is not None for p in x.path)
        no_nullified_cond = x.wav.shape[-1] > 1  # we don't want to read from cache when condition dropout
        if self.wav_cache is not None and no_undefined_paths and no_nullified_cond:
            paths = [Path(p) for p in x.path if p is not None]
            embed = self.wav_cache.get_embed_from_cache(paths, x)
        else:
            embed = self._compute_wav_embedding(x.wav, x.length, x.sample_rate, reduce_mean=True)
        if self.normalize:
            embed = torch.nn.functional.normalize(embed, p=2.0, dim=-1)
        return embed

    def tokenize(self, x: JointEmbedCondition) -> JointEmbedCondition:
        # Trying to limit as much as possible sync points when the cache is warm.
        no_undefined_paths = all(p is not None for p in x.path)
        if self.wav_cache is not None and no_undefined_paths:
            assert all([p is not None for p in x.path]), "Cache requires all JointEmbedCondition paths to be provided"
            paths = [Path(p) for p in x.path if p is not None]
            self.wav_cache.populate_embed_cache(paths, x)
        if self.text_cache is not None and no_undefined_paths:
            assert all([p is not None for p in x.path]), "Cache requires all JointEmbedCondition paths to be provided"
            paths = [Path(p) for p in x.path if p is not None]
            self.text_cache.populate_embed_cache(paths, x)
        return x

    def _get_embed(self, x: JointEmbedCondition) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Extract shared latent representation from either the wav or the text using CLAP."""
        # decide whether to use text embedding at train time or not
        use_text_embed = random.random() < self.text_p
        if self.training and not use_text_embed:
            embed = self._get_wav_embedding(x)
            empty_idx = torch.LongTensor([])  # we assume we always have the audio wav
        else:
            embed = self._get_text_embedding(x)
            empty_idx = torch.LongTensor([i for i, xi in enumerate(x.text) if xi is None or xi == ""])
        return embed, empty_idx


def dropout_condition(sample: ConditioningAttributes, condition_type: str, condition: str) -> ConditioningAttributes:
    """Utility function for nullifying an attribute inside an ConditioningAttributes object.
    If the condition is of type "wav", then nullify it using `nullify_condition` function.
    If the condition is of any other type, set its value to None.
    Works in-place.
    """
    if condition_type not in ['text', 'wav', 'joint_embed']:
        raise ValueError(
            "dropout_condition got an unexpected condition type!"
            f" expected 'text', 'wav' or 'joint_embed' but got '{condition_type}'"
        )

    if condition not in getattr(sample, condition_type):
        raise ValueError(
            "dropout_condition received an unexpected condition!"
            f" expected wav={sample.wav.keys()} and text={sample.text.keys()}"
            f" but got '{condition}' of type '{condition_type}'!"
        )

    if condition_type == 'wav':
        wav_cond = sample.wav[condition]
        sample.wav[condition] = nullify_wav(wav_cond)
    elif condition_type == 'joint_embed':
        embed = sample.joint_embed[condition]
        sample.joint_embed[condition] = nullify_joint_embed(embed)
    else:
        sample.text[condition] = None

    return sample


class DropoutModule(nn.Module):
    """Base module for all dropout modules."""
    def __init__(self, seed: int = 1234):
        super().__init__()
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)


class AttributeDropout(DropoutModule):
    """Dropout with a given probability per attribute.
    This is different from the behavior of ClassifierFreeGuidanceDropout as this allows for attributes
    to be dropped out separately. For example, "artist" can be dropped while "genre" remains.
    This is in contrast to ClassifierFreeGuidanceDropout where if "artist" is dropped "genre"
    must also be dropped.

    Args:
        p (tp.Dict[str, float]): A dict mapping between attributes and dropout probability. For example:
            ...
            "genre": 0.1,
            "artist": 0.5,
            "wav": 0.25,
            ...
        active_on_eval (bool, optional): Whether the dropout is active at eval. Default to False.
        seed (int, optional): Random seed.
    """
    def __init__(self, p: tp.Dict[str, tp.Dict[str, float]], active_on_eval: bool = False, seed: int = 1234):
        super().__init__(seed=seed)
        self.active_on_eval = active_on_eval
        # construct dict that return the values from p otherwise 0
        self.p = {}
        for condition_type, probs in p.items():
            self.p[condition_type] = defaultdict(lambda: 0, probs)

    def forward(self, samples: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
        """
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after certain attributes were set to None.
        """
        if not self.training and not self.active_on_eval:
            return samples

        samples = deepcopy(samples)
        for condition_type, ps in self.p.items():  # for condition types [text, wav]
            for condition, p in ps.items():  # for attributes of each type (e.g., [artist, genre])
                if torch.rand(1, generator=self.rng).item() < p:
                    for sample in samples:
                        dropout_condition(sample, condition_type, condition)
        return samples

    def __repr__(self):
        return f"AttributeDropout({dict(self.p)})"


class ClassifierFreeGuidanceDropout(DropoutModule):
    """Classifier Free Guidance dropout.
    All attributes are dropped with the same probability.

    Args:
        p (float): Probability to apply condition dropout during training.
        seed (int): Random seed.
    """
    def __init__(self, p: float, seed: int = 1234):
        super().__init__(seed=seed)
        self.p = p

    def forward(self, samples: tp.List[ConditioningAttributes]) -> tp.List[ConditioningAttributes]:
        """
        Args:
            samples (list[ConditioningAttributes]): List of conditions.
        Returns:
            list[ConditioningAttributes]: List of conditions after all attributes were set to None.
        """
        if not self.training:
            return samples

        # decide on which attributes to drop in a batched fashion
        drop = torch.rand(1, generator=self.rng).item() < self.p
        if not drop:
            return samples

        # nullify conditions of all attributes
        samples = deepcopy(samples)
        for condition_type in ["wav", "text"]:
            for sample in samples:
                for condition in sample.attributes[condition_type]:
                    dropout_condition(sample, condition_type, condition)
        return samples

    def __repr__(self):
        return f"ClassifierFreeGuidanceDropout(p={self.p})"


class ConditioningProvider(nn.Module):
    """Prepare and provide conditions given all the supported conditioners.

    Args:
        conditioners (dict): Dictionary of conditioners.
        device (torch.device or str, optional): Device for conditioners and output condition types.
    """
    def __init__(self, conditioners: tp.Dict[str, BaseConditioner], device: tp.Union[torch.device, str] = "cpu"):
        super().__init__()
        self.device = device
        self.conditioners = nn.ModuleDict(conditioners)

    @property
    def joint_embed_conditions(self):
        return [m.attribute for m in self.conditioners.values() if isinstance(m, JointEmbeddingConditioner)]

    @property
    def has_joint_embed_conditions(self):
        return len(self.joint_embed_conditions) > 0

    @property
    def text_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, TextConditioner)]

    @property
    def wav_conditions(self):
        return [k for k, v in self.conditioners.items() if isinstance(v, WaveformConditioner)]

    @property
    def has_wav_condition(self):
        return len(self.wav_conditions) > 0

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
        joint_embeds = self._collate_joint_embeds(inputs)

        assert set(text.keys() | wavs.keys() | joint_embeds.keys()).issubset(set(self.conditioners.keys())), (
            f"Got an unexpected attribute! Expected {self.conditioners.keys()}, ",
            f"got {text.keys(), wavs.keys(), joint_embeds.keys()}"
        )

        for attribute, batch in chain(text.items(), wavs.items(), joint_embeds.items()):
            output[attribute] = self.conditioners[attribute].tokenize(batch)
        return output

    def forward(self, tokenized: tp.Dict[str, tp.Any]) -> tp.Dict[str, ConditionType]:
        """Compute pairs of `(embedding, mask)` using the configured conditioners and the tokenized representations.
        The output is for example:
        {
            "genre": (torch.Tensor([B, 1, D_genre]), torch.Tensor([B, 1])),
            "description": (torch.Tensor([B, T_desc, D_desc]), torch.Tensor([B, T_desc])),
            ...
        }

        Args:
            tokenized (dict): Dict of tokenized representations as returned by `tokenize()`.
        """
        output = {}
        for attribute, inputs in tokenized.items():
            condition, mask = self.conditioners[attribute](inputs)
            output[attribute] = (condition, mask)
        return output

    def _collate_text(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, tp.List[tp.Optional[str]]]:
        """Given a list of ConditioningAttributes objects, compile a dictionary where the keys
        are the attributes and the values are the aggregated input per attribute.
        For example:
        Input:
        [
            ConditioningAttributes(text={"genre": "Rock", "description": "A rock song with a guitar solo"}, wav=...),
            ConditioningAttributes(text={"genre": "Hip-hop", "description": "A hip-hop verse"}, wav=...),
        ]
        Output:
        {
            "genre": ["Rock", "Hip-hop"],
            "description": ["A rock song with a guitar solo", "A hip-hop verse"]
        }

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, list[str, optional]]: A dictionary mapping an attribute name to text batch.
        """
        out: tp.Dict[str, tp.List[tp.Optional[str]]] = defaultdict(list)
        texts = [x.text for x in samples]
        for text in texts:
            for condition in self.text_conditions:
                out[condition].append(text[condition])
        return out

    def _collate_wavs(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, WavCondition]:
        """Generate a dict where the keys are attributes by which we fetch similar wavs,
        and the values are Tensors of wavs according to said attributes.

        *Note*: by the time the samples reach this function, each sample should have some waveform
        inside the "wav" attribute. It should be either:
        1. A real waveform
        2. A null waveform due to the sample having no similar waveforms (nullified by the dataset)
        3. A null waveform due to it being dropped in a dropout module (nullified by dropout)

        Args:
            samples (list of ConditioningAttributes): List of ConditioningAttributes samples.
        Returns:
            dict[str, WavCondition]: A dictionary mapping an attribute name to wavs.
        """
        wavs = defaultdict(list)
        lengths = defaultdict(list)
        sample_rates = defaultdict(list)
        paths = defaultdict(list)
        seek_times = defaultdict(list)
        out: tp.Dict[str, WavCondition] = {}

        for sample in samples:
            for attribute in self.wav_conditions:
                wav, length, sample_rate, path, seek_time = sample.wav[attribute]
                assert wav.dim() == 3, f"Got wav with dim={wav.dim()}, but expected 3 [1, C, T]"
                assert wav.size(0) == 1, f"Got wav [B, C, T] with shape={wav.shape}, but expected B == 1"
                # mono-channel conditioning
                wav = wav.mean(1, keepdim=True)  # [1, 1, T]
                wavs[attribute].append(wav.flatten())  # [T]
                lengths[attribute].append(length)
                sample_rates[attribute].extend(sample_rate)
                paths[attribute].extend(path)
                seek_times[attribute].extend(seek_time)

        # stack all wavs to a single tensor
        for attribute in self.wav_conditions:
            stacked_wav, _ = collate(wavs[attribute], dim=0)
            out[attribute] = WavCondition(
                stacked_wav.unsqueeze(1), torch.cat(lengths[attribute]), sample_rates[attribute],
                paths[attribute], seek_times[attribute])

        return out

    def _collate_joint_embeds(self, samples: tp.List[ConditioningAttributes]) -> tp.Dict[str, JointEmbedCondition]:
        """Generate a dict where the keys are attributes by which we compute joint embeddings,
        and the values are Tensors of pre-computed embeddings and the corresponding text attributes.

        Args:
            samples (list[ConditioningAttributes]): List of ConditioningAttributes samples.
        Returns:
            A dictionary mapping an attribute name to joint embeddings.
        """
        texts = defaultdict(list)
        wavs = defaultdict(list)
        lengths = defaultdict(list)
        sample_rates = defaultdict(list)
        paths = defaultdict(list)
        seek_times = defaultdict(list)
        channels: int = 0

        out = {}
        for sample in samples:
            for attribute in self.joint_embed_conditions:
                wav, text, length, sample_rate, path, seek_time = sample.joint_embed[attribute]
                assert wav.dim() == 3
                if channels == 0:
                    channels = wav.size(1)
                else:
                    assert channels == wav.size(1), "not all audio has same number of channels in batch"
                assert wav.size(0) == 1, "Expecting single-wav batch in the collate method"
                wav = einops.rearrange(wav, "b c t -> (b c t)")  # [1, C, T] => [C * T]
                wavs[attribute].append(wav)
                texts[attribute].extend(text)
                lengths[attribute].append(length)
                sample_rates[attribute].extend(sample_rate)
                paths[attribute].extend(path)
                seek_times[attribute].extend(seek_time)

        for attribute in self.joint_embed_conditions:
            stacked_texts = texts[attribute]
            stacked_paths = paths[attribute]
            stacked_seek_times = seek_times[attribute]
            stacked_wavs = pad_sequence(wavs[attribute]).to(self.device)
            stacked_wavs = einops.rearrange(stacked_wavs, "(c t) b -> b c t", c=channels)
            stacked_sample_rates = sample_rates[attribute]
            stacked_lengths = torch.cat(lengths[attribute]).to(self.device)
            assert stacked_lengths.size(0) == stacked_wavs.size(0)
            assert len(stacked_sample_rates) == stacked_wavs.size(0)
            assert len(stacked_texts) == stacked_wavs.size(0)
            out[attribute] = JointEmbedCondition(
                text=stacked_texts, wav=stacked_wavs,
                length=stacked_lengths, sample_rate=stacked_sample_rates,
                path=stacked_paths, seek_time=stacked_seek_times)

        return out


class ConditionFuser(StreamingModule):
    """Condition fuser handles the logic to combine the different conditions
    to the actual model input.

    Args:
        fuse2cond (tp.Dict[str, str]): A dictionary that says how to fuse
            each condition. For example:
            {
                "prepend": ["description"],
                "sum": ["genre", "bpm"],
                "cross": ["description"],
            }
        cross_attention_pos_emb (bool, optional): Use positional embeddings in cross attention.
        cross_attention_pos_emb_scale (int): Scale for positional embeddings in cross attention if used.
    """
    FUSING_METHODS = ["sum", "prepend", "cross", "input_interpolate"]

    def __init__(self, fuse2cond: tp.Dict[str, tp.List[str]], cross_attention_pos_emb: bool = False,
                 cross_attention_pos_emb_scale: float = 1.0):
        super().__init__()
        assert all(
            [k in self.FUSING_METHODS for k in fuse2cond.keys()]
        ), f"Got invalid fuse method, allowed methods: {self.FUSING_METHODS}"
        self.cross_attention_pos_emb = cross_attention_pos_emb
        self.cross_attention_pos_emb_scale = cross_attention_pos_emb_scale
        self.fuse2cond: tp.Dict[str, tp.List[str]] = fuse2cond
        self.cond2fuse: tp.Dict[str, str] = {}
        for fuse_method, conditions in fuse2cond.items():
            for condition in conditions:
                self.cond2fuse[condition] = fuse_method

    def forward(
        self,
        input: torch.Tensor,
        conditions: tp.Dict[str, ConditionType]
    ) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Fuse the conditions to the provided model input.

        Args:
            input (torch.Tensor): Transformer input.
            conditions (dict[str, ConditionType]): Dict of conditions.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: The first tensor is the transformer input
                after the conditions have been fused. The second output tensor is the tensor
                used for cross-attention or None if no cross attention inputs exist.
        """
        B, T, _ = input.shape

        if 'offsets' in self._streaming_state:
            first_step = False
            offsets = self._streaming_state['offsets']
        else:
            first_step = True
            offsets = torch.zeros(input.shape[0], dtype=torch.long, device=input.device)

        assert set(conditions.keys()).issubset(set(self.cond2fuse.keys())), \
            f"given conditions contain unknown attributes for fuser, " \
            f"expected {self.cond2fuse.keys()}, got {conditions.keys()}"
        cross_attention_output = None
        for cond_type, (cond, cond_mask) in conditions.items():
            op = self.cond2fuse[cond_type]
            if op == 'sum':
                input += cond
            elif op == 'input_interpolate':
                cond = einops.rearrange(cond, "b t d -> b d t")
                cond = F.interpolate(cond, size=input.shape[1])
                input += einops.rearrange(cond, "b d t -> b t d")
            elif op == 'prepend':
                if first_step:
                    input = torch.cat([cond, input], dim=1)
            elif op == 'cross':
                if cross_attention_output is not None:
                    cross_attention_output = torch.cat([cross_attention_output, cond], dim=1)
                else:
                    cross_attention_output = cond
            else:
                raise ValueError(f"unknown op ({op})")

        if self.cross_attention_pos_emb and cross_attention_output is not None:
            positions = torch.arange(
                cross_attention_output.shape[1],
                device=cross_attention_output.device
            ).view(1, -1, 1)
            pos_emb = create_sin_embedding(positions, cross_attention_output.shape[-1])
            cross_attention_output = cross_attention_output + self.cross_attention_pos_emb_scale * pos_emb

        if self._is_streaming:
            self._streaming_state['offsets'] = offsets + T

        return input, cross_attention_output
