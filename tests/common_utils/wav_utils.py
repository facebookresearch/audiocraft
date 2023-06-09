# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import typing as tp

import torch
import torchaudio


def get_white_noise(chs: int = 1, num_frames: int = 1):
    wav = torch.randn(chs, num_frames)
    return wav


def get_batch_white_noise(bs: int = 1, chs: int = 1, num_frames: int = 1):
    wav = torch.randn(bs, chs, num_frames)
    return wav


def save_wav(path: str, wav: torch.Tensor, sample_rate: int):
    fp = Path(path)
    kwargs: tp.Dict[str, tp.Any] = {}
    if fp.suffix == '.wav':
        kwargs['encoding'] = 'PCM_S'
        kwargs['bits_per_sample'] = 16
    elif fp.suffix == '.mp3':
        kwargs['compression'] = 320
    torchaudio.save(str(fp), wav, sample_rate, **kwargs)
