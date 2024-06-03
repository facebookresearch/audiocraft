# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import typing as tp
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from audiocraft.models.loaders import load_audioseal_models


class WMModel(ABC, nn.Module):
    """
    A wrapper interface to different watermarking models for
    training or evaluation purporses
    """

    @abstractmethod
    def get_watermark(
        self,
        x: torch.Tensor,
        message: tp.Optional[torch.Tensor] = None,
        sample_rate: int = 16_000,
    ) -> torch.Tensor:
        """Get the watermark from an audio tensor and a message.
        If the input message is None, a random message of
        n bits {0,1} will be generated
        """

    @abstractmethod
    def detect_watermark(self, x: torch.Tensor) -> torch.Tensor:
        """Detect the watermarks from the audio signal

        Args:
            x: Audio signal, size batch x frames

        Returns:
            tensor of size (B, 2+n, frames) where:
            Detection results of shape (B, 2, frames)
            Message decoding results of shape (B, n, frames)
        """


class AudioSeal(WMModel):
    """Wrap Audioseal (https://github.com/facebookresearch/audioseal) for the
    training and evaluation. The generator and detector are jointly trained
    """

    def __init__(
        self,
        generator: nn.Module,
        detector: nn.Module,
        nbits: int = 0,
    ):
        super().__init__()
        self.generator = generator  # type: ignore
        self.detector = detector  # type: ignore

        # Allow to re-train an n-bit model with new 0-bit message
        self.nbits = nbits if nbits else self.generator.msg_processor.nbits

    def get_watermark(
        self,
        x: torch.Tensor,
        message: tp.Optional[torch.Tensor] = None,
        sample_rate: int = 16_000,
    ) -> torch.Tensor:
        return self.generator.get_watermark(x, message=message, sample_rate=sample_rate)

    def detect_watermark(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect the watermarks from the audio signal.  The first two units of the output
        are used for detection, the rest is used to decode the message. If the audio is
        not watermarked, the message will be random.

        Args:
            x: Audio signal, size batch x frames
        Returns
            torch.Tensor: Detection + decoding results of shape (B, 2+nbits, T).
        """

        # Getting the direct decoded message from the detector
        result = self.detector.detector(x)  # b x 2+nbits
        # hardcode softmax on 2 first units used for detection
        result[:, :2, :] = torch.softmax(result[:, :2, :], dim=1)
        return result

    def forward(  # generator
        self,
        x: torch.Tensor,
        message: tp.Optional[torch.Tensor] = None,
        sample_rate: int = 16_000,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """Apply the watermarking to the audio signal x with a tune-down ratio (default 1.0)"""
        wm = self.get_watermark(x, message)
        return x + alpha * wm

    @staticmethod
    def get_pretrained(name="base", device=None) -> WMModel:
        if device is None:
            if torch.cuda.device_count():
                device = "cuda"
            else:
                device = "cpu"
        return load_audioseal_models("facebook/audioseal", filename=name, device=device)
