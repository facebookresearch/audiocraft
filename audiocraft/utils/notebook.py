# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

try:
    import IPython.display as ipd  # type: ignore
except ImportError:
    # Note in a notebook...
    pass


import torch


def display_audio(samples: torch.Tensor, sample_rate: int):
    """Renders an audio player for the given audio samples.

    Args:
        samples (torch.Tensor): a Tensor of decoded audio samples
            with shapes [B, C, T] or [C, T]
        sample_rate (int): sample rate audio should be displayed with.
    """
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for audio in samples:
        ipd.display(ipd.Audio(audio, rate=sample_rate))
