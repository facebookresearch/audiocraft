# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import typing as tp

import julius
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torchaudio.functional.filtering import highpass_biquad, treble_biquad


def basic_loudness(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """This is a simpler loudness function that is more stable.
    Args:
        waveform(torch.Tensor): audio waveform of dimension `(..., channels, time)`
        sample_rate (int): sampling rate of the waveform
    Returns:
        loudness loss as a scalar
    """

    if waveform.size(-2) > 5:
        raise ValueError("Only up to 5 channels are supported.")
    eps = torch.finfo(torch.float32).eps
    gate_duration = 0.4
    overlap = 0.75
    gate_samples = int(round(gate_duration * sample_rate))
    step = int(round(gate_samples * (1 - overlap)))

    # Apply K-weighting
    waveform = treble_biquad(waveform, sample_rate, 4.0, 1500.0, 1 / math.sqrt(2))
    waveform = highpass_biquad(waveform, sample_rate, 38.0, 0.5)

    # Compute the energy for each block
    energy = torch.square(waveform).unfold(-1, gate_samples, step)
    energy = torch.mean(energy, dim=-1)

    # Compute channel-weighted summation
    g = torch.tensor([1.0, 1.0, 1.0, 1.41, 1.41], dtype=waveform.dtype, device=waveform.device)
    g = g[: energy.size(-2)]

    energy_weighted = torch.sum(g.unsqueeze(-1) * energy, dim=-2)
    # loudness with epsilon for stability. Not as much precision in the very low loudness sections
    loudness = -0.691 + 10 * torch.log10(energy_weighted + eps)
    return loudness


def _unfold(a: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """Given input of size [*OT, T], output Tensor of size [*OT, F, K]
    with K the kernel size, by extracting frames with the given stride.
    This will pad the input so that `F = ceil(T / K)`.
    see https://github.com/pytorch/pytorch/issues/60466
    """
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1, "data should be contiguous"
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)


class FLoudnessRatio(nn.Module):
    """FSNR loss.

    Input should be [B, C, T], output is scalar.

    Args:
        sample_rate (int): Sample rate.
        segment (float or None): Evaluate on chunks of that many seconds. If None, evaluate on
            entire audio only.
        overlap (float): Overlap between chunks, i.e. 0.5 = 50 % overlap.
        epsilon (float): Epsilon value for numerical stability.
        n_bands (int): number of mel scale bands that we include
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        segment: tp.Optional[float] = 20,
        overlap: float = 0.5,
        epsilon: float = torch.finfo(torch.float32).eps,
        n_bands: int = 0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment = segment
        self.overlap = overlap
        self.epsilon = epsilon
        if n_bands == 0:
            self.filter = None
        else:
            self.filter = julius.SplitBands(sample_rate=sample_rate, n_bands=n_bands)
        self.loudness = torchaudio.transforms.Loudness(sample_rate)

    def forward(self, out_sig: torch.Tensor, ref_sig: torch.Tensor) -> torch.Tensor:
        B, C, T = ref_sig.shape
        assert ref_sig.shape == out_sig.shape
        assert self.filter is not None
        bands_ref = self.filter(ref_sig)
        bands_out = self.filter(out_sig)
        l_noise = self.loudness(bands_ref - bands_out)
        l_ref = self.loudness(bands_ref)
        l_ratio = (l_noise - l_ref).view(-1, B)
        loss = torch.nn.functional.softmax(l_ratio, dim=0) * l_ratio
        return loss.sum()


class TLoudnessRatio(nn.Module):
    """TSNR loss.

    Input should be [B, C, T], output is scalar.

    Args:
        sample_rate (int): Sample rate.
        segment (float or None): Evaluate on chunks of that many seconds. If None, evaluate on
            entire audio only.
        overlap (float): Overlap between chunks, i.e. 0.5 = 50 % overlap.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        segment: float = 0.5,
        overlap: float = 0.5,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment = segment
        self.overlap = overlap
        self.loudness = torchaudio.transforms.Loudness(sample_rate)

    def forward(self, out_sig: torch.Tensor, ref_sig: torch.Tensor) -> torch.Tensor:
        B, C, T = ref_sig.shape
        assert ref_sig.shape == out_sig.shape
        assert C == 1

        frame = int(self.segment * self.sample_rate)
        stride = int(frame * (1 - self.overlap))
        gt = _unfold(ref_sig, frame, stride).view(-1, 1, frame)
        est = _unfold(out_sig, frame, stride).view(-1, 1, frame)
        l_noise = self.loudness(gt - est)  # watermark
        l_ref = self.loudness(gt)  # ground truth
        l_ratio = (l_noise - l_ref).view(-1, B)
        loss = torch.nn.functional.softmax(l_ratio, dim=0) * l_ratio
        return loss.sum()


class TFLoudnessRatio(nn.Module):
    """TF-loudness ratio loss.

    Input should be [B, C, T], output is scalar.

    Args:
        sample_rate (int): Sample rate.
        segment (float or None): Evaluate on chunks of that many seconds. If None, evaluate on
            entire audio only.
        overlap (float): Overlap between chunks, i.e. 0.5 = 50 % overlap.
        n_bands (int): number of bands to separate
        temperature (float): temperature of the softmax step
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        segment: float = 0.5,
        overlap: float = 0.5,
        n_bands: int = 0,
        clip_min: float = -100,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment = segment
        self.overlap = overlap
        self.clip_min = clip_min
        self.temperature = temperature
        if n_bands == 0:
            self.filter = None
        else:
            self.n_bands = n_bands
            self.filter = julius.SplitBands(sample_rate=sample_rate, n_bands=n_bands)

    def forward(self, out_sig: torch.Tensor, ref_sig: torch.Tensor) -> torch.Tensor:
        B, C, T = ref_sig.shape

        assert ref_sig.shape == out_sig.shape
        assert C == 1
        assert self.filter is not None

        bands_ref = self.filter(ref_sig).view(B * self.n_bands, 1, -1)
        bands_out = self.filter(out_sig).view(B * self.n_bands, 1, -1)
        frame = int(self.segment * self.sample_rate)
        stride = int(frame * (1 - self.overlap))
        gt = _unfold(bands_ref, frame, stride).squeeze(1).contiguous().view(-1, 1, frame)
        est = _unfold(bands_out, frame, stride).squeeze(1).contiguous().view(-1, 1, frame)
        l_noise = basic_loudness(est - gt, sample_rate=self.sample_rate)  # watermark
        l_ref = basic_loudness(gt, sample_rate=self.sample_rate)  # ground truth
        l_ratio = (l_noise - l_ref).view(-1, B)
        loss = torch.nn.functional.softmax(l_ratio / self.temperature, dim=0) * l_ratio
        return loss.mean()
