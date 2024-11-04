# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main model for using MusicGen-Stem. This will combine all the required components
and provide easy access to the generation API.
"""

import typing as tp
import warnings

import torch

from .encodec import CompressionModel, MultiStemCompressionModel
from .genmodel import BaseGenModel
from .lm import LMModel
from .loaders import load_compression_model, load_lm_model
from ..data.audio_utils import convert_audio
from ..modules.conditioners import ConditioningAttributes, WavCondition
from demucs.apply import apply_model

MelodyList = tp.List[tp.Optional[torch.Tensor]]
MelodyType = tp.Union[torch.Tensor, MelodyList]


class MusicGenStem(BaseGenModel):
    """MusicGen main model with convenient generation API.

    Args:
        name (str): name of the model.
        compression_model (CompressionModel): Compression model
            used to map audio to invertible discrete representations.
        lm (LMModel): Language model over discrete representations.
        max_duration (float, optional): maximum duration the model can produce,
            otherwise, inferred from the training params.
    """
    def __init__(self, name: str, compression_model: CompressionModel, lm: LMModel,
                 max_duration: tp.Optional[float] = None):
        super().__init__(name, compression_model, lm, max_duration)
        self.set_generation_params(duration=15)  # default duration
        self.demucs_is_inited = False


    @staticmethod
    def get_pretrained(name: str = 'facebook/musicgen-stem-6cb', device=None):
        """Return pretrained model, we provide four models:
        - facebook/musicgen-stem-6cb (1.5B), text to music with 6 codebooks
            1 for the bass, 1 for the drums and 4 for the other
          # see: https://huggingface.co/facebook/musicgen-stem-6cb
        - facebook/musicgen-stem-7cb (1.5B), text to music with 7 codebooks
            2 for the bass, 1 for the drums and 4 for the other
          # see: https://huggingface.co/facebook/musicgen-stem-7cb
        """
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        lm = load_lm_model(name, device=device)
        sources = ['bass', 'drums', 'other']
        all_compression_models = [
            load_compression_model(file_or_url_or_id=name, device=device, filename='compression_state_dict_bass.bin'),
            load_compression_model(file_or_url_or_id=name, device=device, filename='compression_state_dict_drums.bin'),
            load_compression_model(file_or_url_or_id=name, device=device, filename='compression_state_dict_other.bin'),
            ]
        compression_model = MultiStemCompressionModel(sources, all_compression_models)

        return MusicGenStem(name, compression_model, lm)

    def set_generation_params(self, use_sampling: bool = True, top_k: int = 250,
                              top_p: float = 0.0, temperature: float = 1.0,
                              duration: float = 30.0, cfg_coef: float = 3.0,
                              two_step_cfg: bool = False, extend_stride: float = 18):
        """Set the generation parameters for MusicGen.

        Args:
            use_sampling (bool, optional): Use sampling if True, else do argmax decoding. Defaults to True.
            top_k (int, optional): top_k used for sampling. Defaults to 250.
            top_p (float, optional): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.0.
            temperature (float, optional): Softmax temperature parameter. Defaults to 1.0.
            duration (float, optional): Duration of the generated waveform. Defaults to 30.0.
            cfg_coef (float, optional): Coefficient used for classifier free guidance. Defaults to 3.0.
            two_step_cfg (bool, optional): If True, performs 2 forward for Classifier Free Guidance,
                instead of batching together the two. This has some impact on how things
                are padded but seems to have little impact in practice.
            extend_stride: when doing extended generation (i.e. more than 30 seconds), by how much
                should we extend the audio each time. Larger values will mean less context is
                preserved, and shorter value will require extra computations.
        """
        assert extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
        self.extend_stride = extend_stride
        self.duration = duration
        self.generation_params = {
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef,
            'two_step_cfg': two_step_cfg,
        }

    @torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            descriptions: tp.Sequence[tp.Optional[str]],
            prompt: tp.Optional[torch.Tensor],
            melody_wavs: tp.Optional[MelodyList] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (torch.Tensor, optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        attributes = [
            ConditioningAttributes(text={'description': description})
            for description in descriptions]

        if melody_wavs is None:
            for attr in attributes:
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    sample_rate=[self.sample_rate],
                    path=[None])
        else:
            if 'self_wav' not in self.lm.condition_provider.conditioners:
                raise RuntimeError("This model doesn't support melody conditioning. "
                                   "Use the `melody` model.")
            assert len(melody_wavs) == len(descriptions), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}"
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    attr.wav['self_wav'] = WavCondition(
                        torch.zeros((1, 1, 1), device=self.device),
                        torch.tensor([0], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None])
                else:
                    attr.wav['self_wav'] = WavCondition(
                        melody[None].to(device=self.device),
                        torch.tensor([melody.shape[-1]], device=self.device),
                        sample_rate=[self.sample_rate],
                        path=[None],
                    )

        if prompt is not None:
            if descriptions is not None:
                assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to(self.device)
            prompt_tokens = self.compression_model.encode(prompt)
        else:
            prompt_tokens = None
        return attributes, prompt_tokens

    def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False,
                         stems_prompt: tp.Optional[torch.Tensor] = None, which_stems: tp.Optional[tp.List[int]] = None,
                         ) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (list of ConditioningAttributes): Conditions used for generation (text/melody).
            prompt_tokens (torch.Tensor, optional): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        total_gen_len = int(self.duration * self.frame_rate)
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0

        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(generated_tokens, tokens_to_generate)
            else:
                print(f'{generated_tokens: 6d} / {tokens_to_generate: 6d}', end='\r')

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = None
        if progress:
            callback = _progress_callback

        if self.duration <= self.max_duration:
            # generate by sampling from LM, simple case.
            with self.autocast:
                gen_tokens = self.lm.generate(
                    prompt_tokens, attributes,
                    callback=callback, max_gen_len=total_gen_len, 
                    stems_prompt=stems_prompt, which_stems=which_stems, **self.generation_params)

        else:
            # now this gets a bit messier, we need to handle prompts,
            # melody conditioning etc.
            ref_wavs = [attr.wav['self_wav'] for attr in attributes]
            all_tokens = []
            if prompt_tokens is None:
                prompt_length = 0
            else:
                all_tokens.append(prompt_tokens)
                prompt_length = prompt_tokens.shape[-1]

            assert self.extend_stride is not None, "Stride should be defined to generate beyond max_duration"
            assert self.extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
            stride_tokens = int(self.frame_rate * self.extend_stride)

            while current_gen_offset + prompt_length < total_gen_len:
                time_offset = current_gen_offset / self.frame_rate
                chunk_duration = min(self.duration - time_offset, self.max_duration)
                max_gen_len = int(chunk_duration * self.frame_rate)
                for attr, ref_wav in zip(attributes, ref_wavs):
                    wav_length = ref_wav.length.item()
                    if wav_length == 0:
                        continue
                    # We will extend the wav periodically if it not long enough.
                    # we have to do it here rather than in conditioners.py as otherwise
                    # we wouldn't have the full wav.
                    initial_position = int(time_offset * self.sample_rate)
                    wav_target_length = int(self.max_duration * self.sample_rate)
                    positions = torch.arange(initial_position,
                                             initial_position + wav_target_length, device=self.device)
                    attr.wav['self_wav'] = WavCondition(
                        ref_wav[0][..., positions % wav_length],
                        torch.full_like(ref_wav[1], wav_target_length),
                        [self.sample_rate] * ref_wav[0].size(0),
                        [None], [0.])
                with self.autocast:
                    gen_tokens = self.lm.generate(
                        prompt_tokens, attributes,
                        callback=callback, max_gen_len=max_gen_len, **self.generation_params)
                if prompt_tokens is None:
                    all_tokens.append(gen_tokens)
                else:
                    all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
                prompt_tokens = gen_tokens[:, :, stride_tokens:]
                prompt_length = prompt_tokens.shape[-1]
                current_gen_offset += stride_tokens

            gen_tokens = torch.cat(all_tokens, dim=-1)
        return gen_tokens

    def generate_with_prompt_codes(self, descriptions: tp.List[str], prompt_codes: torch.Tensor, progress: bool = False,
                              return_tokens: bool = False, silence_tokens: tp.List[int] = [870, 530, 621, 1215], 
                              silence_prompt: bool = False):
        assert prompt_codes is not None
        attributes, _ = self._prepare_tokens_and_attributes(descriptions, None)
        prompt_tokens = prompt_codes
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if silence_prompt:
            for i in range(tokens.shape[1]):  # Loop over columns (second dimension)
                for b in range(tokens.shape[0]):
                    mask = (tokens[b, i, :] == 2048)  # Create a mask for elements equal to target_value
                    tokens[b, i, mask] = silence_tokens[i]  # Replace these elements
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)


    def regenerate_from_codes(self, codes: torch.Tensor, which_codes: tp.List[int], descriptions: tp.List[str], 
                              prompt: torch.Tensor = None, prompt_sample_rate: int = None, prompt_codes: torch.Tensor = None, 
                              progress: bool = False, 
                              return_tokens: bool = False, silence_tokens: tp.List[int] = [870, 530, 621, 1215], 
                              silence_prompt: bool = False) \
            -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """
        Given a stream of codes, regenerate the desired streams.

        Args:
            codes: must be [1, K, T] (K being the number of codebooks)
            which_codes: list e.g. [0, 2, 3] if you want to keep these index of streams
            descriptions: list of strings of length B. 
            prompt: tensor [1, 1, T] of waveform if we want a few seconds prompt before generating continuation
                with stems fixed by codes and which codes 

        Outputs:
            regenerated songs with these descriptions
        """
        assert (prompt is None) or (prompt_codes is None)
        if prompt is not None:
            prompt = self._prepare_wav_for_compression_model(prompt, prompt_sample_rate)
            prompt = prompt.expand(len(descriptions), -1, -1, -1)

        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, prompt)
        if prompt_codes is not None:
            prompt_tokens = prompt_codes
            prompt_tokens = prompt_tokens.expand(len(descriptions), -1, -1)
        tokens = self._generate_tokens(attributes, prompt_tokens, progress, stems_prompt=codes.expand(len(descriptions), -1, -1), which_stems=which_codes)
        if silence_prompt:
            for i in range(tokens.shape[1]):  # Loop over columns (second dimension)
                for b in range(tokens.shape[0]):
                    mask = (tokens[b, i, :] == 2048)  # Create a mask for elements equal to target_value
                    tokens[b, i, mask] = silence_tokens[i]  # Replace these elements
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    def _init_demucs(self):
        from demucs import pretrained
        self.__dict__['demucs'] = pretrained.get_model('htdemucs').to(self.device)
        demucs_sources: list = self.demucs.sources  # type: ignore
        sources = self.compression_model.sources
        self.stem_indices = torch.LongTensor([demucs_sources.index(source) for source in sources]).to(self.device)
        self.demucs_is_inited = True


    def regenerate_from_wav(self, wav: torch.Tensor, sample_rate: int, 
                            which_codes: tp.List[int], descriptions: tp.List[str], 
                            prompt: torch.Tensor = None, prompt_sample_rate: int = None, prompt_codes: torch.Tensor = None,
                            progress: bool = False, return_tokens: bool = False, silence_tokens: tp.List[int] = [870, 530, 621, 1215], 
                              silence_prompt: bool = False) \
            -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        stems = self._prepare_wav_for_compression_model(wav, sample_rate)
        codes = self.compression_model.encode(stems)
        return self.regenerate_from_codes(codes=codes, which_codes=which_codes, descriptions=descriptions, prompt=prompt,
                                          prompt_sample_rate=prompt_sample_rate, prompt_codes=prompt_codes, 
                                          progress=progress, return_tokens=return_tokens, silence_tokens=silence_tokens, silence_prompt=silence_prompt)
        
    def _prepare_wav_for_compression_model(self, wav, sample_rate):
        if not self.demucs_is_inited:
            self._init_demucs()
            print("demucs is inited")
        wav = wav.to(self.device)
        wav = convert_audio(
            wav, sample_rate, self.demucs.samplerate, self.demucs.audio_channels)  # type: ignore
        stems = apply_model(self.demucs, wav, device=self.device)
        stems = stems[:, self.stem_indices]
        stems = convert_audio(stems, self.demucs.samplerate, self.sample_rate, 1)  # type: ignore
        return stems

    def generate_with_style(self, descriptions: tp.List[str], bdo_wavs: MelodyType,
                             progress: bool = False,
                             return_tokens: bool = False) -> tp.Union[torch.Tensor,
                                                                      tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on text and melody.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            bdo_wavs: (torch.Tensor or list of Tensor): A batch of waveforms used as
                melody conditioning. Should have shape [B, 3, T] with B matching the description length.
                It can be [3, T] if there is a single description. It can also be
                a list of [3, T] tensors.
            melody_sample_rate: (int): Sample rate of the melody waveforms.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if isinstance(bdo_wavs, torch.Tensor):
            if bdo_wavs.dim() == 2:
                bdo_wavs = bdo_wavs[None]
            if bdo_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            assert bdo_wavs.shape[1] == 3, "should input 3 conditions"
            bdo_wavs = list(bdo_wavs)
        else:
            for melody in bdo_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."
                    assert melody.shape[0] == 3, "should input 3 conditions"

        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions, prompt=None,
                                                                        melody_wavs=bdo_wavs)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)