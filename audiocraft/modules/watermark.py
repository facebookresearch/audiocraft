# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import random

import torch


def pad(
    x_wm: torch.Tensor, central: bool = False
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """Pad a watermarked signal at the begining and the end

    Args:
        x_wm (torch.Tensor) : watermarked audio
        central (bool): Whether to mask the middle of the wave (around 34%) or the two tails
            (beginning and ending frames)

    Returns:
        padded (torch.Tensor): padded signal
        true_predictions(torch.Tensor): A binary mask where 1 represents
        watermarked and 0 represents non-watermarked."""
    # keep at leat 34% of watermarked signal
    max_start = int(0.33 * x_wm.size(-1))
    min_end = int(0.66 * x_wm.size(-1))
    starts = torch.randint(0, max_start, size=(x_wm.size(0),))
    ends = torch.randint(min_end, x_wm.size(-1), size=(x_wm.size(0),))
    mask = torch.zeros_like(x_wm)
    for i in range(x_wm.size(0)):
        mask[i, :, starts[i]: ends[i]] = 1
    if central:
        mask = 1 - mask
    padded = x_wm * mask
    true_predictions = torch.cat([1 - mask, mask], dim=1)
    return padded, true_predictions


def mix(
    x: torch.Tensor, x_wm: torch.Tensor, window_size: float = 0.5, shuffle: bool = False
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    """
    Mixes a window of the non-watermarked audio signal 'x' into the watermarked audio signal 'x_wm'.

    This function takes two tensors of shape [batch, channels, frames], copies a window of 'x' with the specified
    'window_size' into 'x_wm', and returns a new tensor that is a mix between the watermarked (1 - mix_percent %)
    and non-watermarked audio (mix_percent %).

    Args:
        x (torch.Tensor): The non-watermarked audio signal tensor.
        x_wm (torch.Tensor): The watermarked audio signal tensor.
        window_size (float, optional): The percentage of 'x' to copy into 'x_wm' (between 0 and 1).
        shuffle (bool): whether or no keep the mix from the same batch element

    Returns:
        tuple: A tuple containing two tensors:
            - mixed_tensor (torch.Tensor): The resulting mixed audio signal tensor.
            - mask (torch.Tensor): A binary mask where 1 represents watermarked and 0 represents non-watermarked.

    Raises:
        AssertionError: If 'window_size' is not between 0 and 1.
    """
    assert 0 < window_size <= 1, "window_size should be between 0 and 1"

    # Calculate the maximum starting point for the window
    max_start_point = x.shape[-1] - int(window_size * x.shape[-1])

    # Generate a random starting point within the adjusted valid range
    start_point = random.randint(0, max_start_point)

    # Calculate the window size in frames
    total_frames = x.shape[-1]
    window_frames = int(window_size * total_frames)

    # Create a mask tensor to identify watermarked and non-watermarked portions
    # it outputs two classes to match the detector output shape of [bsz, 2, frames]
    # Copy the random window from 'x' to 'x_wm'
    mixed = x_wm.detach().clone()

    true_predictions = torch.cat(
        [torch.zeros_like(mixed), torch.ones_like(mixed)], dim=1
    )
    # non-watermark class correct labels.
    true_predictions[:, 0, start_point: start_point + window_frames] = 1.0
    # watermarked class correct labels
    true_predictions[:, 1, start_point: start_point + window_frames] = 0.0

    if shuffle:
        # Take the middle part from a random element of the batch
        shuffle_idx = torch.randint(0, x.size(0), (x.size(0),))
        mixed[:, :, start_point: start_point + window_frames] = x[shuffle_idx][
            :, :, start_point: start_point + window_frames
        ]
    else:
        mixed[:, :, start_point: start_point + window_frames] = x[
            :, :, start_point: start_point + window_frames
        ]

    return mixed, true_predictions
