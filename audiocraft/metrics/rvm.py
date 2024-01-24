# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import torch
from torch import nn
import torchaudio


def db_to_scale(volume: tp.Union[float, torch.Tensor]):
    return 10 ** (volume / 20)


def scale_to_db(scale: torch.Tensor, min_volume: float = -120):
    min_scale = db_to_scale(min_volume)
    return 20 * torch.log10(scale.clamp(min=min_scale))


class RelativeVolumeMel(nn.Module):
    """Relative volume melspectrogram measure.

    Computes a measure of distance over two mel spectrogram that is interpretable in terms
    of decibels. Given `x_ref` and `x_est` two waveforms of shape `[*, T]`, it will
    first renormalize both by the ground truth of `x_ref`.

    ..Warning:: This class returns the volume of the distortion at the spectrogram level,
        e.g. low negative values reflects lower distortion levels. For a SNR (like reported
        in the MultiBandDiffusion paper), just take `-rvm`.

    Then it computes the mel spectrogram `z_ref` and `z_est` and compute volume of the difference
    relative to the volume of `z_ref` for each time-frequency bin. It further adds some limits, e.g.
    clamping the values between -25 and 25 dB (controlled by `min_relative_volume` and `max_relative_volume`)
    with the goal of avoiding the loss being dominated by parts where the reference is almost silent.
    Indeed, volumes in dB can take unbounded values both towards -oo and +oo, which can make the final
    average metric harder to interpret. Besides, anything below -30 dB of attenuation would sound extremely
    good (for a neural network output, although sound engineers typically aim for much lower attenuations).
    Similarly, anything above +30 dB would just be completely missing the target, and there is no point
    in measuring by exactly how much it missed it. -25, 25 is a more conservative range, but also more
    in line with what neural nets currently can achieve.

    For instance, a Relative Volume Mel (RVM) score of -10 dB means that on average, the delta between
    the target and reference mel-spec is 10 dB lower than the reference mel-spec value.

    The metric can be aggregated over a given frequency band in order have different insights for
    different region of the spectrum. `num_aggregated_bands` controls the number of bands.

    ..Warning:: While this function is optimized for interpretability, nothing was done to ensure it
        is numerically stable when computing its gradient. We thus advise against using it as a training loss.

    Args:
        sample_rate (int): Sample rate of the input audio.
        n_mels (int): Number of mel bands to use.
        n_fft (int): Number of frequency bins for the STFT.
        hop_length (int): Hop length of the STFT and the mel-spectrogram.
        min_relative_volume (float): The error `z_ref - z_est` volume is given relative to
            the volume of `z_ref`. If error is smaller than -25 dB of `z_ref`, then it is clamped.
        max_relative_volume (float): Same as `min_relative_volume` but clamping if the error is larger than that.
        max_initial_gain (float): When rescaling the audio at the very beginning, we will limit the gain
            to that amount, to avoid rescaling near silence. Given in dB.
        min_activity_volume (float): When computing the reference level from `z_ref`, will clamp low volume
            bins to that amount. This is effectively our "zero" level for the reference mel-spectrogram,
            and anything below that will be considered equally.
        num_aggregated_bands (int): Number of bands to keep when computing the average RVM value.
            For instance, a value of 3 would give 3 scores, roughly for low, mid and high freqs.
    """
    def __init__(self, sample_rate: int = 24000, n_mels: int = 80, n_fft: int = 512,
                 hop_length: int = 128, min_relative_volume: float = -25,
                 max_relative_volume: float = 25, max_initial_gain: float = 25,
                 min_activity_volume: float = -25,
                 num_aggregated_bands: int = 4) -> None:
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
            normalized=True, sample_rate=sample_rate, power=2)
        self.min_relative_volume = min_relative_volume
        self.max_relative_volume = max_relative_volume
        self.max_initial_gain = max_initial_gain
        self.min_activity_volume = min_activity_volume
        self.num_aggregated_bands = num_aggregated_bands

    def forward(self, estimate: torch.Tensor, ground_truth: torch.Tensor) -> tp.Dict[str, torch.Tensor]:
        """Compute RVM metric between estimate and reference samples.

        Args:
            estimate (torch.Tensor): Estimate sample.
            ground_truth (torch.Tensor): Reference sample.

        Returns:
            dict[str, torch.Tensor]: Metrics with keys `rvm` for the overall average, and `rvm_{k}`
            for the RVM over the k-th band (k=0..num_aggregated_bands - 1).
        """
        min_scale = db_to_scale(-self.max_initial_gain)
        std = ground_truth.pow(2).mean().sqrt().clamp(min=min_scale)
        z_gt = self.melspec(ground_truth / std).sqrt()
        z_est = self.melspec(estimate / std).sqrt()

        delta = z_gt - z_est
        ref_db = scale_to_db(z_gt, self.min_activity_volume)
        delta_db = scale_to_db(delta.abs(), min_volume=-120)
        relative_db = (delta_db - ref_db).clamp(self.min_relative_volume, self.max_relative_volume)
        dims = list(range(relative_db.dim()))
        dims.remove(dims[-2])
        losses_per_band = relative_db.mean(dim=dims)
        aggregated = [chunk.mean() for chunk in losses_per_band.chunk(self.num_aggregated_bands, dim=0)]
        metrics = {f'rvm_{index}': value for index, value in enumerate(aggregated)}
        metrics['rvm'] = losses_per_band.mean()
        return metrics
