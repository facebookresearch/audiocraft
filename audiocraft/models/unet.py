# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Pytorch Unet Module used for diffusion.
"""

from dataclasses import dataclass
import typing as tp

import torch
from torch import nn
from torch.nn import functional as F
from audiocraft.modules.transformer import StreamingTransformer, create_sin_embedding


@dataclass
class Output:
    sample: torch.Tensor


def get_model(cfg, channels: int, side: int, num_steps: int):
    if cfg.model == 'unet':
        return DiffusionUnet(
            chin=channels, num_steps=num_steps, **cfg.diffusion_unet)
    else:
        raise RuntimeError('Not Implemented')


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, norm_groups: int = 4,
                 dilation: int = 1, activation: tp.Type[nn.Module] = nn.ReLU,
                 dropout: float = 0.):
        super().__init__()
        stride = 1
        padding = dilation * (kernel - stride) // 2
        Conv = nn.Conv1d
        Drop = nn.Dropout1d
        self.norm1 = nn.GroupNorm(norm_groups, channels)
        self.conv1 = Conv(channels, channels, kernel, 1, padding, dilation=dilation)
        self.activation1 = activation()
        self.dropout1 = Drop(dropout)

        self.norm2 = nn.GroupNorm(norm_groups, channels)
        self.conv2 = Conv(channels, channels, kernel, 1, padding, dilation=dilation)
        self.activation2 = activation()
        self.dropout2 = Drop(dropout)

    def forward(self, x):
        h = self.dropout1(self.conv1(self.activation1(self.norm1(x))))
        h = self.dropout2(self.conv2(self.activation2(self.norm2(h))))
        return x + h


class DecoderLayer(nn.Module):
    def __init__(self, chin: int, chout: int, kernel: int = 4, stride: int = 2,
                 norm_groups: int = 4, res_blocks: int = 1, activation: tp.Type[nn.Module] = nn.ReLU,
                 dropout: float = 0.):
        super().__init__()
        padding = (kernel - stride) // 2
        self.res_blocks = nn.Sequential(
            *[ResBlock(chin, norm_groups=norm_groups, dilation=2**idx, dropout=dropout)
              for idx in range(res_blocks)])
        self.norm = nn.GroupNorm(norm_groups, chin)
        ConvTr = nn.ConvTranspose1d
        self.convtr = ConvTr(chin, chout, kernel, stride, padding, bias=False)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_blocks(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.convtr(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, chin: int, chout: int, kernel: int = 4, stride: int = 2,
                 norm_groups: int = 4, res_blocks: int = 1, activation: tp.Type[nn.Module] = nn.ReLU,
                 dropout: float = 0.):
        super().__init__()
        padding = (kernel - stride) // 2
        Conv = nn.Conv1d
        self.conv = Conv(chin, chout, kernel, stride, padding, bias=False)
        self.norm = nn.GroupNorm(norm_groups, chout)
        self.activation = activation()
        self.res_blocks = nn.Sequential(
            *[ResBlock(chout, norm_groups=norm_groups, dilation=2**idx, dropout=dropout)
              for idx in range(res_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        stride, = self.conv.stride
        pad = (stride - (T % stride)) % stride
        x = F.pad(x, (0, pad))

        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.res_blocks(x)
        return x


class BLSTM(nn.Module):
    """BiLSTM with same hidden units as input dim.
    """
    def __init__(self, dim, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        return x


class DiffusionUnet(nn.Module):
    def __init__(self, chin: int = 3, hidden: int = 24, depth: int = 3, growth: float = 2.,
                 max_channels: int = 10_000, num_steps: int = 1000, emb_all_layers=False, cross_attention: bool = False,
                 bilstm: bool = False, transformer: bool = False,
                 codec_dim: tp.Optional[int] = None, **kwargs):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.embeddings: tp.Optional[nn.ModuleList] = None
        self.embedding = nn.Embedding(num_steps, hidden)
        if emb_all_layers:
            self.embeddings = nn.ModuleList()
        self.condition_embedding: tp.Optional[nn.Module] = None
        for d in range(depth):
            encoder = EncoderLayer(chin, hidden, **kwargs)
            decoder = DecoderLayer(hidden, chin, **kwargs)
            self.encoders.append(encoder)
            self.decoders.insert(0, decoder)
            if emb_all_layers and d > 0:
                assert self.embeddings is not None
                self.embeddings.append(nn.Embedding(num_steps, hidden))
            chin = hidden
            hidden = min(int(chin * growth), max_channels)
        self.bilstm: tp.Optional[nn.Module]
        if bilstm:
            self.bilstm = BLSTM(chin)
        else:
            self.bilstm = None
        self.use_transformer = transformer
        self.cross_attention = False
        if transformer:
            self.cross_attention = cross_attention
            self.transformer = StreamingTransformer(chin, 8, 6, bias_ff=False, bias_attn=False,
                                                    cross_attention=cross_attention)

        self.use_codec = False
        if codec_dim is not None:
            self.conv_codec = nn.Conv1d(codec_dim, chin, 1)
            self.use_codec = True

    def forward(self, x: torch.Tensor, step: tp.Union[int, torch.Tensor], condition: tp.Optional[torch.Tensor] = None):
        skips = []
        bs = x.size(0)
        z = x
        view_args = [1]
        if type(step) is torch.Tensor:
            step_tensor = step
        else:
            step_tensor = torch.tensor([step], device=x.device, dtype=torch.long).expand(bs)

        for idx, encoder in enumerate(self.encoders):
            z = encoder(z)
            if idx == 0:
                z = z + self.embedding(step_tensor).view(bs, -1, *view_args).expand_as(z)
            elif self.embeddings is not None:
                z = z + self.embeddings[idx - 1](step_tensor).view(bs, -1, *view_args).expand_as(z)

            skips.append(z)

        if self.use_codec:  # insert condition in the bottleneck
            assert condition is not None, "Model defined for conditionnal generation"
            condition_emb = self.conv_codec(condition)  # reshape to the bottleneck dim
            assert condition_emb.size(-1) <= 2 * z.size(-1), \
                f"You are downsampling the conditionning with factor >=2 : {condition_emb.size(-1)=} and {z.size(-1)=}"
            if not self.cross_attention:

                condition_emb = torch.nn.functional.interpolate(condition_emb, z.size(-1))
                assert z.size() == condition_emb.size()
                z += condition_emb
                cross_attention_src = None
            else:
                cross_attention_src = condition_emb.permute(0, 2, 1)  # B, T, C
                B, T, C = cross_attention_src.shape
                positions = torch.arange(T, device=x.device).view(1, -1, 1)
                pos_emb = create_sin_embedding(positions, C, max_period=10_000, dtype=cross_attention_src.dtype)
                cross_attention_src = cross_attention_src + pos_emb
        if self.use_transformer:
            z = self.transformer(z.permute(0, 2, 1), cross_attention_src=cross_attention_src).permute(0, 2, 1)
        else:
            if self.bilstm is None:
                z = torch.zeros_like(z)
            else:
                z = self.bilstm(z)

        for decoder in self.decoders:
            s = skips.pop(-1)
            z = z[:, :, :s.shape[2]]
            z = z + s
            z = decoder(z)

        z = z[:, :, :x.shape[2]]
        return Output(z)
