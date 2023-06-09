# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from torch import nn
import torch


class XPos(nn.Module):
    """Length-extrapolatable positional embedding (xPos) from [Sun et al 2022](https://arxiv.org/abs/2212.10554v1).
    This applies an exponential decay to the RoPE rotation matrix.

    Args:
        dim (int): Embedding dimension.
        smoothing (float): Smoothing factor applied to the decay rates.
        base_scale (int): Base decay rate, given in terms of scaling time.
        device (torch.device or None): Device on which to initialize the module.
        dtype (torch.dtype): dtype to use to generate the embedding.
    """
    def __init__(self, dim: int, smoothing: float = 0.4, base_scale: int = 512,
                 device=None, dtype: torch.dtype = torch.float32):
        super().__init__()
        assert dim % 2 == 0
        assert dtype in [torch.float64, torch.float32]
        self.dtype = dtype
        self.base_scale = base_scale

        half_dim = dim // 2
        adim = torch.arange(half_dim, device=device, dtype=dtype)
        decay_rates = (adim / half_dim + smoothing) / (1.0 + smoothing)
        self.register_buffer("decay_rates", decay_rates)
        self.decay: tp.Optional[torch.Tensor] = None

    def get_decay(self, start: int, end: int):
        """Create complex decay tensor, cache values for fast computation.
        """
        if self.decay is None or end > self.decay.shape[0]:
            assert isinstance(self.decay_rates, torch.Tensor)  # Satisfy type checker.
            idx = torch.arange(end, device=self.decay_rates.device, dtype=self.dtype)
            power = idx / self.base_scale
            scale = self.decay_rates ** power.unsqueeze(-1)
            self.decay = torch.polar(scale, torch.zeros_like(scale))
        return self.decay[start:end]  # [T, C/2]


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (RoPE) from [Su et al 2022](https://arxiv.org/abs/2104.09864).

    Args:
        dim (int): Embedding dimension (twice the number of frequencies).
        max_period (float): Maximum period of the rotation frequencies.
        xpos (bool): Use xPos, applies an exponential decay to rotation matrix.
        scale (float): Scale of positional embedding, set to 0 to deactivate.
        device (torch.device or None): Device on which to initialize the module.
        dtype (torch.dtype): dtype to use to generate the embedding.
    """
    def __init__(self, dim: int, max_period: float = 10000.0, xpos: bool = False,
                 scale: float = 1.0, device=None, dtype: torch.dtype = torch.float32):
        super().__init__()
        assert dim % 2 == 0
        self.scale = scale
        assert dtype in [torch.float64, torch.float32]
        self.dtype = dtype

        adim = torch.arange(0, dim, 2, device=device, dtype=dtype)[: (dim // 2)]
        frequencies = 1.0 / (max_period ** (adim / dim))
        self.register_buffer("frequencies", frequencies)
        self.rotation: tp.Optional[torch.Tensor] = None

        self.xpos = XPos(dim, device=device, dtype=dtype) if xpos else None

    def get_rotation(self, start: int, end: int):
        """Create complex rotation tensor, cache values for fast computation.
        """
        if self.rotation is None or end > self.rotation.shape[0]:
            assert isinstance(self.frequencies, torch.Tensor)  # Satisfy type checker.
            idx = torch.arange(end, device=self.frequencies.device, dtype=self.dtype)
            angles = torch.outer(idx, self.frequencies)
            self.rotation = torch.polar(torch.ones_like(angles), angles)
        return self.rotation[start:end]

    def rotate(self, x: torch.Tensor, start: int = 0, invert_decay: bool = False):
        """Apply rope rotation to query or key tensor.
        """
        T = x.shape[1]
        rotation = self.get_rotation(start, start + T).unsqueeze(0).unsqueeze(2)

        if self.xpos:
            decay = self.xpos.get_decay(start, start + T).unsqueeze(0).unsqueeze(2)
        else:
            decay = 1.0

        if invert_decay:
            decay = decay ** -1

        x_complex = torch.view_as_complex(x.to(self.dtype).reshape(*x.shape[:-1], -1, 2))
        scaled_rotation = (rotation * decay) * self.scale + (1.0 - self.scale)
        x_out = torch.view_as_real(x_complex * scaled_rotation).flatten(-2)

        return x_out.type_as(x)

    def rotate_qk(self, query: torch.Tensor, key: torch.Tensor, start: int = 0):
        """ Apply rope rotation to both query and key tensors.
        Supports streaming mode, in which query and key are not expected to have the same shape.
        In streaming mode, key will be of legnth [P + C] with P the cached past timesteps, but
        query will be [C] (typically C == 1).

        Args:
            query (torch.Tensor): Query to rotate.
            key (torch.Tensor): Key to rotate.
            start (int): Start index of the sequence for time offset.
        """
        query_timesteps = query.shape[1]
        key_timesteps = key.shape[1]
        streaming_offset = key_timesteps - query_timesteps

        query_out = self.rotate(query, start + streaming_offset)
        key_out = self.rotate(key, start, invert_decay=True)

        return query_out, key_out
