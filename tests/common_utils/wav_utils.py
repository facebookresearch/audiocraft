# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch

from audiocraft.data.audio import audio_write


def get_white_noise(chs: int = 1, num_frames: int = 1):
    wav = torch.randn(chs, num_frames)
    return wav


def get_batch_white_noise(bs: int = 1, chs: int = 1, num_frames: int = 1):
    wav = torch.randn(bs, chs, num_frames)
    return wav


def save_wav(path: str, wav: torch.Tensor, sample_rate: int):
    assert wav.dim() == 2, wav.shape
    fp = Path(path)
    assert fp.suffix in ['.mp3', '.ogg', '.wav', '.flac'], fp
    audio_write(fp.parent / fp.stem, wav, sample_rate, fp.suffix[1:],
                normalize=False, strategy='clip', peak_clip_headroom_db=0)
