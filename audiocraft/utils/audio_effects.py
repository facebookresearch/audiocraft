# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import inspect
import random
import typing as tp
from functools import partial

import julius
import omegaconf
import torch
from julius import fft_conv1d, resample_frac

from ..data.audio_utils import get_aac, get_mp3

if tp.TYPE_CHECKING:
    from ..models.encodec import CompressionModel


def select_audio_effects(
    audio_effects: tp.Dict,
    weights: tp.Optional[tp.Dict] = None,
    mode: str = "all",
    max_length: tp.Optional[int] = None,
):
    """Samples a subset of audio effects methods from the `AudioEffects` class.

    This function allows you to select a subset of audio effects
    based on the chosen selection mode and optional weights.

    Args:
        audio_effects (dict): A dictionary of available audio augmentations, usually
            obtained from the output of the 'get_audio_effects' function.
        weights (dict): A dictionary mapping augmentation names to their corresponding
            probabilities of being selected. This argument is used when 'mode' is set
            to "weighted." If 'weights' is None, all augmentations have equal
            probability of being selected.
        mode (str): The selection mode, which can be one of the following:
            - "all": Select all available augmentations.
            - "weighted": Select augmentations based on their probabilities in the
              'weights' dictionary.
        max_length (int): The maximum number of augmentations to select. If 'max_length'
            is None, no limit is applied.

    Returns:
        dict: A subset of the 'audio_effects' dictionary containing the selected audio
        augmentations.

    Note:
        - In "all" mode, all available augmentations are selected.
        - In "weighted" mode, augmentations are selected with a probability
          proportional to their weights specified in the 'weights' dictionary.
        - If 'max_length' is set, the function limits the number of selected
          augmentations.
        - If no augmentations are selected or 'audio_effects' is empty, the function
          defaults to including an "identity" augmentation.
        - The "identity" augmentation means that no audio effect is applied.
    """
    if mode == "all":  # original code
        out = audio_effects
    elif mode == "weighted":
        # Probability proportionnal to weights
        assert weights is not None
        out = {
            name: value
            for name, value in audio_effects.items()
            if random.random() < weights.get(name, 1.0)
        }
    else:
        raise ValueError(f"Unknown mode {mode}")
    if max_length is not None:
        # Help having a deterministic limit of the gpu memory usage
        random_keys = random.sample(list(out.keys()), max_length)
        out = {key: out[key] for key in random_keys}
    if len(out) == 0:  # Check not to return empty dict
        out = {"identity": AudioEffects.identity}
    return out


def get_audio_effects(cfg: omegaconf.DictConfig):
    """Automatically pull the list all effects available in this class based on the parameters from the cfg

    Returns:
        dict: A dict of names and pointers to all methods in this class.
    """
    assert hasattr(cfg, "audio_effects")
    cfg_audio_effects = dict(cfg["audio_effects"])
    return {
        name: partial(value, **cfg_audio_effects.get(name, {}))
        for name, value in inspect.getmembers(AudioEffects)
        if inspect.isfunction(value)
    }


def audio_effect_return(
    tensor: torch.Tensor, mask: tp.Optional[torch.Tensor]
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the mask if it was in the input otherwise only the output tensor"""
    if mask is None:
        return tensor
    else:
        return tensor, mask


def generate_pink_noise(length: int) -> torch.Tensor:
    """Generate pink noise using Voss-McCartney algorithm with PyTorch."""
    num_rows = 16
    array = torch.randn(num_rows, length // num_rows + 1)
    reshaped_array = torch.cumsum(array, dim=1)
    reshaped_array = reshaped_array.reshape(-1)
    reshaped_array = reshaped_array[:length]
    # Normalize
    pink_noise = reshaped_array / torch.max(torch.abs(reshaped_array))
    return pink_noise


def compress_with_encodec(
    tensor: torch.Tensor,
    n_q: int,
    model: "CompressionModel",
    sample_rate: int,
    mask: tp.Optional[torch.Tensor] = None,
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Special augmentation function that compresses and decompresses wav tensor
    using a compression model with the n_q codebooks
    """

    model.to(tensor.device)
    model.set_num_codebooks(n_q)
    codes, scale = model.encode(
        julius.resample_frac(tensor, old_sr=sample_rate, new_sr=model.sample_rate)
    )
    compressed = model.decode(codes=codes, scale=scale)
    return audio_effect_return(
        tensor=julius.resample_frac(
            compressed, old_sr=model.sample_rate, new_sr=sample_rate
        ),
        mask=mask,
    )


def apply_compression_skip_grad(tensor: torch.Tensor, compression_fn, **kwargs):
    """Applies a specified compression function to the audio tensor.
    Whire carrying over the grads to the output tensor with skip through estimator
    this is a straight through estimator to make mp3/aac compression differentiable
    see more: Yin et al. 2019 https://arxiv.org/pdf/1903.05662.pdf

    Args:
        tensor (torch.Tensor): The input audio tensor.
        compression_fn (function): The compression function to apply.
        **kwargs: Additional keyword arguments for the compression function.

    Returns:
        torch.Tensor: The output tensor after applying compression and straight through estimator.
    """
    compressed = compression_fn(tensor.detach(), **kwargs)

    # Trim compressed output if needed
    compressed = compressed[:, :, : tensor.size(-1)]

    # Straight through estimator for differentiable compression
    out = tensor + (compressed - tensor).detach()

    # Check that gradients are not broken
    if out.requires_grad:
        assert (
            out.grad_fn
        ), "The computation graph might be broken due to compression augmentation."

    return out


class AudioEffects:
    @staticmethod
    def speed(
        tensor: torch.Tensor,
        speed_range: tuple = (0.5, 1.5),
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Function to change the speed of a batch of audio data.
        The output will have a different length !

        Args:
            audio_batch (torch.Tensor): The batch of audio data in torch tensor format.
            speed (float): The speed to change the audio to.

        Returns:
            torch.Tensor: The batch of audio data with the speed changed.
        """
        speed = torch.FloatTensor(1).uniform_(*speed_range)
        new_sr = int(sample_rate * 1 / speed)
        resampled_tensor = julius.resample.resample_frac(tensor, sample_rate, new_sr)
        if mask is None:
            return resampled_tensor
        else:
            return resampled_tensor, torch.nn.functional.interpolate(
                mask, size=resampled_tensor.size(-1), mode="nearest-exact"
            )

    @staticmethod
    def updownresample(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        intermediate_freq: int = 32000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

        orig_shape = tensor.shape
        # upsample
        tensor = resample_frac(tensor, sample_rate, intermediate_freq)
        # downsample
        tensor = resample_frac(tensor, intermediate_freq, sample_rate)

        assert tensor.shape == orig_shape
        return audio_effect_return(tensor=tensor, mask=mask)

    @staticmethod
    def echo(
        tensor: torch.Tensor,
        volume_range: tuple = (0.1, 0.5),
        duration_range: tuple = (0.1, 0.5),
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Attenuating the audio volume by a factor of 0.4, delaying it by 100ms,
        and then overlaying it with the original.

        Args:
            tensor: 3D Tensor representing the audio signal [bsz, channels, frames]
            volumne range: volume range of the echo signal
            duration range: duration range of the echo signal
            sample_rate: Sample rate of the audio signal.
        Returns:
            Audio signal with reverb.
        """

        # Create a simple impulse response
        # Duration of the impulse response in seconds
        duration = torch.FloatTensor(1).uniform_(*duration_range)
        volume = torch.FloatTensor(1).uniform_(*volume_range)

        n_samples = int(sample_rate * duration)
        impulse_response = torch.zeros(n_samples).type(tensor.type()).to(tensor.device)

        # Define a few reflections with decreasing amplitude
        impulse_response[0] = 1.0  # Direct sound

        impulse_response[
            int(sample_rate * duration) - 1
        ] = volume  # First reflection after 100ms

        # Add batch and channel dimensions to the impulse response
        impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)

        # Convolve the audio signal with the impulse response
        reverbed_signal = fft_conv1d(tensor, impulse_response)

        # Normalize to the original amplitude range for stability
        reverbed_signal = (
            reverbed_signal
            / torch.max(torch.abs(reverbed_signal))
            * torch.max(torch.abs(tensor))
        )

        # Ensure tensor size is not changed
        tmp = torch.zeros_like(tensor)
        tmp[..., : reverbed_signal.shape[-1]] = reverbed_signal
        reverbed_signal = tmp

        return audio_effect_return(tensor=reverbed_signal, mask=mask)

    @staticmethod
    def random_noise(
        waveform: torch.Tensor,
        noise_std: float = 0.001,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Add Gaussian noise to the waveform."""
        noise = torch.randn_like(waveform) * noise_std
        noisy_waveform = waveform + noise
        return audio_effect_return(tensor=noisy_waveform, mask=mask)

    @staticmethod
    def pink_noise(
        waveform: torch.Tensor,
        noise_std: float = 0.01,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Add pink background noise to the waveform."""
        noise = generate_pink_noise(waveform.shape[-1]) * noise_std
        noise = noise.to(waveform.device)
        # Assuming waveform is of shape (bsz, channels, length)
        noisy_waveform = waveform + noise.unsqueeze(0).unsqueeze(0).to(waveform.device)
        return audio_effect_return(tensor=noisy_waveform, mask=mask)

    @staticmethod
    def lowpass_filter(
        waveform: torch.Tensor,
        cutoff_freq: float = 5000,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Filter the lowpass frequency from the waveform"""
        return audio_effect_return(
            tensor=julius.lowpass_filter(waveform, cutoff=cutoff_freq / sample_rate),
            mask=mask,
        )

    @staticmethod
    def highpass_filter(
        waveform: torch.Tensor,
        cutoff_freq: float = 500,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Filter the highpass frequency from the waveform"""
        return audio_effect_return(
            tensor=julius.highpass_filter(waveform, cutoff=cutoff_freq / sample_rate),
            mask=mask,
        )

    @staticmethod
    def bandpass_filter(
        waveform: torch.Tensor,
        cutoff_freq_low: float = 300,
        cutoff_freq_high: float = 8000,
        sample_rate: int = 16000,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Apply a bandpass filter to the waveform by cascading
        a high-pass filter followed by a low-pass filter.

        Args:
            waveform (torch.Tensor): Input audio waveform.
            low_cutoff (float): Lower cutoff frequency.
            high_cutoff (float): Higher cutoff frequency.
            sample_rate (int): The sample rate of the waveform.

        Returns:
            torch.Tensor: Filtered audio waveform.
        """

        return audio_effect_return(
            tensor=julius.bandpass_filter(
                waveform,
                cutoff_low=cutoff_freq_low / sample_rate,
                cutoff_high=cutoff_freq_high / sample_rate,
            ),
            mask=mask,
        )

    @staticmethod
    def smooth(
        tensor: torch.Tensor,
        window_size_range: tuple = (2, 10),
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Smooths the input tensor (audio signal) using a moving average filter with the
        given window size.

        Args:
            tensor (torch.Tensor): Input audio tensor. Assumes tensor shape is (batch_size,
            channels, time).
            window_size (int): Size of the moving average window.
            mask: Masks for the input wave

        Returns:
            torch.Tensor: Smoothed audio tensor.
        """

        window_size = int(torch.FloatTensor(1).uniform_(*window_size_range))
        # Create a uniform smoothing kernel
        kernel = torch.ones(1, 1, window_size).type(tensor.type()) / window_size
        kernel = kernel.to(tensor.device)

        smoothed = fft_conv1d(tensor, kernel)
        # Ensure tensor size is not changed
        tmp = torch.zeros_like(tensor)
        tmp[..., : smoothed.shape[-1]] = smoothed
        smoothed = tmp

        return audio_effect_return(tensor=smoothed, mask=mask)

    @staticmethod
    def boost_audio(
        tensor: torch.Tensor,
        amount: float = 20,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Filter the lowpass frequency from the waveform"""
        return audio_effect_return(tensor=tensor * (1 + amount / 100), mask=mask)

    @staticmethod
    def duck_audio(
        tensor: torch.Tensor,
        amount: float = 20,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Mask input wav with some ducked signnals"""
        return audio_effect_return(tensor=tensor * (1 - amount / 100), mask=mask)

    @staticmethod
    def identity(
        tensor: torch.Tensor, mask: tp.Optional[torch.Tensor] = None
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return audio_effect_return(tensor=tensor, mask=mask)

    @staticmethod
    def mp3_compression(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        bitrate: str = "128k",
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Compress audio using MP3 algorithm
        Args:
            tensor (torch.Tensor): The input audio tensor.
            sample_rate (int): The sample rate of the audio.
            bitrate (str): The bitrate for MP3 compression.

        Returns:
            torch.Tensor: The output tensor after applying MP3 compression.
        """
        out = apply_compression_skip_grad(
            tensor, get_mp3, sr=sample_rate, bitrate=bitrate
        )
        return audio_effect_return(tensor=out, mask=mask)

    @staticmethod
    def aac_compression(
        tensor: torch.Tensor,
        sample_rate: int = 16000,
        bitrate: str = "128k",
        lowpass_freq: tp.Optional[int] = None,
        mask: tp.Optional[torch.Tensor] = None,
    ) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Applies AAC compression to an audio tensor.

        Args:
            tensor (torch.Tensor): The input audio tensor.
            sample_rate (int): The sample rate of the audio.
            bitrate (str): The bitrate for AAC compression.
            lowpass_freq (Optional[int]): The frequency for a low-pass filter.

        Returns:
            torch.Tensor: The output tensor after applying AAC compression.
        """
        out = apply_compression_skip_grad(
            tensor, get_aac, sr=sample_rate, bitrate=bitrate, lowpass_freq=lowpass_freq
        )
        return audio_effect_return(tensor=out, mask=mask)
