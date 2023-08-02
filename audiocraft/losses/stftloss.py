# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from MIT code under the original license
# Copyright 2019 Tomoki Hayashi
# MIT License (https://opensource.org/licenses/MIT)
import typing as tp

import torch
from torch import nn
from torch.nn import functional as F


# TODO: Replace with torchaudio.STFT?
def _stft(x: torch.Tensor, fft_size: int, hop_length: int, win_length: int,
          window: tp.Optional[torch.Tensor], normalized: bool) -> torch.Tensor:
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x: Input signal tensor (B, C, T).
        fft_size (int): FFT size.
        hop_length (int): Hop size.
        win_length (int): Window length.
        window (torch.Tensor or None): Window function type.
        normalized (bool): Whether to normalize the STFT or not.

    Returns:
        torch.Tensor: Magnitude spectrogram (B, C, #frames, fft_size // 2 + 1).
    """
    B, C, T = x.shape
    x_stft = torch.stft(
        x.view(-1, T), fft_size, hop_length, win_length, window,
        normalized=normalized, return_complex=True,
    )
    x_stft = x_stft.view(B, C, *x_stft.shape[1:])
    real = x_stft.real
    imag = x_stft.imag

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergenceLoss(nn.Module):
    """Spectral convergence loss.
    """
    def __init__(self, epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor):
        """Calculate forward propagation.

        Args:
            x_mag: Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag: Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            torch.Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + self.epsilon)


class LogSTFTMagnitudeLoss(nn.Module):
    """Log STFT magnitude loss.

    Args:
        epsilon (float): Epsilon value for numerical stability.
    """
    def __init__(self, epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor):
        """Calculate forward propagation.

        Args:
            x_mag (torch.Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (torch.Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            torch.Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(self.epsilon + y_mag), torch.log(self.epsilon + x_mag))


class STFTLosses(nn.Module):
    """STFT losses.

    Args:
        n_fft (int): Size of FFT.
        hop_length (int): Hop length.
        win_length (int): Window length.
        window (str): Window function type.
        normalized (bool): Whether to use normalized STFT or not.
        epsilon (float): Epsilon for numerical stability.
    """
    def __init__(self, n_fft: int = 1024, hop_length: int = 120, win_length: int = 600,
                 window: str = "hann_window", normalized: bool = False,
                 epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergenceLoss(epsilon)
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss(epsilon)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            x (torch.Tensor): Predicted signal (B, T).
            y (torch.Tensor): Groundtruth signal (B, T).
        Returns:
            torch.Tensor: Spectral convergence loss value.
            torch.Tensor: Log STFT magnitude loss value.
        """
        x_mag = _stft(x, self.n_fft, self.hop_length,
                      self.win_length, self.window, self.normalized)  # type: ignore
        y_mag = _stft(y, self.n_fft, self.hop_length,
                      self.win_length, self.window, self.normalized)  # type: ignore
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class STFTLoss(nn.Module):
    """Single Resolution STFT loss.

    Args:
        n_fft (int): Nb of FFT.
        hop_length (int): Hop length.
        win_length (int): Window length.
        window (str): Window function type.
        normalized (bool): Whether to use normalized STFT or not.
        epsilon (float): Epsilon for numerical stability.
        factor_sc (float): Coefficient for the spectral loss.
        factor_mag (float): Coefficient for the magnitude loss.
    """
    def __init__(self, n_fft: int = 1024, hop_length: int = 120, win_length: int = 600,
                 window: str = "hann_window", normalized: bool = False,
                 factor_sc: float = 0.1, factor_mag: float = 0.1,
                 epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        self.loss = STFTLosses(n_fft, hop_length, win_length, window, normalized, epsilon)
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            x (torch.Tensor): Predicted signal (B, T).
            y (torch.Tensor): Groundtruth signal (B, T).
        Returns:
            torch.Tensor: Single resolution STFT loss.
        """
        sc_loss, mag_loss = self.loss(x, y)
        return self.factor_sc * sc_loss + self.factor_mag * mag_loss


class MRSTFTLoss(nn.Module):
    """Multi resolution STFT loss.

    Args:
        n_ffts (Sequence[int]): Sequence of FFT sizes.
        hop_lengths (Sequence[int]): Sequence of hop sizes.
        win_lengths (Sequence[int]): Sequence of window lengths.
        window (str): Window function type.
        factor_sc (float): Coefficient for the spectral loss.
        factor_mag (float): Coefficient for the magnitude loss.
        normalized (bool): Whether to use normalized STFT or not.
        epsilon (float): Epsilon for numerical stability.
    """
    def __init__(self, n_ffts: tp.Sequence[int] = [1024, 2048, 512], hop_lengths: tp.Sequence[int] = [120, 240, 50],
                 win_lengths: tp.Sequence[int] = [600, 1200, 240], window: str = "hann_window",
                 factor_sc: float = 0.1, factor_mag: float = 0.1,
                 normalized: bool = False, epsilon: float = torch.finfo(torch.float32).eps):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(n_ffts, hop_lengths, win_lengths):
            self.stft_losses += [STFTLosses(fs, ss, wl, window, normalized, epsilon)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            x (torch.Tensor): Predicted signal (B, T).
            y (torch.Tensor): Groundtruth signal (B, T).
        Returns:
            torch.Tensor: Multi resolution STFT loss.
        """
        sc_loss = torch.Tensor([0.0])
        mag_loss = torch.Tensor([0.0])
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc * sc_loss + self.factor_mag * mag_loss
