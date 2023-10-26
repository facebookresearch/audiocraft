# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main model for using MusicGen. This will combine all the required components
and provide easy access to the generation API.
"""

import typing as tp
import warnings

import torch

from .encodec import CompressionModel
from .lm import LMModel
from .builders import get_debug_compression_model, get_debug_lm_model
from .loaders import load_compression_model, load_lm_model
from ..data.audio_utils import convert_audio
from ..modules.conditioners import ConditioningAttributes, WavCondition
from ..utils.autocast import TorchAutocast


MelodyList = tp.List[tp.Optional[torch.Tensor]]
MelodyType = tp.Union[torch.Tensor, MelodyList]


# backward compatible names mapping
_HF_MODEL_CHECKPOINTS_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
}


class MusicGen:
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
        self.name = name
        self.compression_model = compression_model
        self.lm = lm
        # Just to be safe, let's put everything in eval mode.
        self.compression_model.eval()
        self.lm.eval()

        if max_duration is None:
            if hasattr(lm, 'cfg'):
                max_duration = lm.cfg.dataset.segment_duration  # type: ignore
            else:
                raise ValueError("You must provide max_duration when building directly MusicGen")
        assert max_duration is not None
        self.max_duration: float = max_duration
        self.device = next(iter(lm.parameters())).device
        self.generation_params: dict = {}
        self.set_generation_params(duration=15)  # 15 seconds by default
        self._progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None
        if self.device.type == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
        else:
            self.autocast = TorchAutocast(
                enabled=True, device_type=self.device.type, dtype=torch.float16)

    @property
    def frame_rate(self) -> float:
        """Roughly the number of AR steps per seconds."""
        return self.compression_model.frame_rate

    @property
    def sample_rate(self) -> int:
        """Sample rate of the generated audio."""
        return self.compression_model.sample_rate

    @property
    def audio_channels(self) -> int:
        """Audio channels of the generated audio."""
        return self.compression_model.channels

    @staticmethod
    def get_pretrained(name: str = 'facebook/musicgen-melody', device=None):
        """Return pretrained model, we provide four models:
        - facebook/musicgen-small (300M), text to music,
          # see: https://huggingface.co/facebook/musicgen-small
        - facebook/musicgen-medium (1.5B), text to music,
          # see: https://huggingface.co/facebook/musicgen-medium
        - facebook/musicgen-melody (1.5B) text to music and text+melody to music,
          # see: https://huggingface.co/facebook/musicgen-melody
        - facebook/musicgen-large (3.3B), text to music,
          # see: https://huggingface.co/facebook/musicgen-large
        """
        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        if name == 'debug':
            # used only for unit tests
            compression_model = get_debug_compression_model(device)
            lm = get_debug_lm_model(device)
            return MusicGen(name, compression_model, lm, max_duration=30)

        if name in _HF_MODEL_CHECKPOINTS_MAP:
            warnings.warn(
                "MusicGen pretrained model relying on deprecated checkpoint mapping. " +
                f"Please use full pre-trained id instead: facebook/musicgen-{name}")
            name = _HF_MODEL_CHECKPOINTS_MAP[name]

        lm = load_lm_model(name, device=device)
        compression_model = load_compression_model(name, device=device)
        if 'self_wav' in lm.condition_provider.conditioners:
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return MusicGen(name, compression_model, lm)

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

    def set_custom_progress_callback(self, progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None):
        """Override the default progress callback."""
        self._progress_callback = progress_callback

    def generate_unconditional(self, num_samples: int, progress: bool = False,
                               return_tokens: bool = False) -> tp.Union[torch.Tensor,
                                                                        tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples in an unconditional manner.

        Args:
            num_samples (int): Number of samples to be generated.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        descriptions: tp.List[tp.Optional[str]] = [None] * num_samples
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    def generate(self, descriptions: tp.List[str], progress: bool = False, return_tokens: bool = False) \
            -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on text.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    def generate_with_chroma(self, descriptions: tp.List[str], melody_wavs: MelodyType,
                             melody_sample_rate: int, progress: bool = False,
                             return_tokens: bool = False) -> tp.Union[torch.Tensor,
                                                                      tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on text and melody.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            melody_wavs: (torch.Tensor or list of Tensor): A batch of waveforms used as
                melody conditioning. Should have shape [B, C, T] with B matching the description length,
                C=1 or 2. It can be [C, T] if there is a single description. It can also be
                a list of [C, T] tensors.
            melody_sample_rate: (int): Sample rate of the melody waveforms.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."

        melody_wavs = [
            convert_audio(wav, melody_sample_rate, self.sample_rate, self.audio_channels)
            if wav is not None else None
            for wav in melody_wavs]
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions, prompt=None,
                                                                        melody_wavs=melody_wavs)
        assert prompt_tokens is None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

    def generate_continuation(self, prompt: torch.Tensor, prompt_sample_rate: int,
                              descriptions: tp.Optional[tp.List[tp.Optional[str]]] = None,
                              progress: bool = False, return_tokens: bool = False) \
            -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on audio prompts.

        Args:
            prompt (torch.Tensor): A batch of waveforms used for continuation.
                Prompt should be [B, C, T], or [C, T] if only one sample is generated.
            prompt_sample_rate (int): Sampling rate of the given audio waveforms.
            descriptions (list of str, optional): A list of strings used as text conditioning. Defaults to None.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if prompt.dim() == 2:
            prompt = prompt[None]
        if prompt.dim() != 3:
            raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
        prompt = convert_audio(prompt, prompt_sample_rate, self.sample_rate, self.audio_channels)
        if descriptions is None:
            descriptions = [None] * len(prompt)
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, prompt)
        assert prompt_tokens is not None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)

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
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens

    def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
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
                self._progress_callback(generated_tokens, total_gen_len)
            else:
                print(f'{generated_tokens: 6d} / {total_gen_len: 6d}', end='\r')

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
                    callback=callback, max_gen_len=total_gen_len, **self.generation_params)

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

    def generate_audio(self, gen_tokens: torch.Tensor):
        """Generate Audio from tokens"""
        assert gen_tokens.dim() == 3
        with torch.no_grad():
            gen_audio = self.compression_model.decode(gen_tokens, None)
        return gen_audio
