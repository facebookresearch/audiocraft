# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from audiocraft.modules.rope import RotaryEmbedding
from audiocraft.modules.transformer import StreamingTransformer, set_efficient_attention_backend


def test_rope():
    set_efficient_attention_backend('torch')
    B, T, H, C = 8, 75, 16, 128

    rope = RotaryEmbedding(dim=C)
    xq = torch.rand((B, T, H, C))
    xk = torch.rand((B, T, H, C))
    xq_out, xk_out = rope.rotate_qk(xq, xk, start=7)

    assert list(xq_out.shape) == [B, T, H, C]
    assert list(xk_out.shape) == [B, T, H, C]


def test_rope_io_dtypes():
    set_efficient_attention_backend('torch')
    B, T, H, C = 8, 75, 16, 128

    rope_32 = RotaryEmbedding(dim=C, dtype=torch.float32)
    rope_64 = RotaryEmbedding(dim=C, dtype=torch.float64)

    # Test bfloat16 inputs w/ both 32 and 64 precision rope.
    xq_16 = torch.rand((B, T, H, C)).to(torch.bfloat16)
    xk_16 = torch.rand((B, T, H, C)).to(torch.bfloat16)
    xq_out, xk_out = rope_32.rotate_qk(xq_16, xk_16)
    assert xq_out.dtype == torch.bfloat16
    xq_out, xk_out = rope_64.rotate_qk(xq_16, xk_16)
    assert xq_out.dtype == torch.bfloat16

    # Test float32 inputs w/ both 32 and 64 precision rope.
    xq_32 = torch.rand((B, T, H, C)).to(torch.float32)
    xk_32 = torch.rand((B, T, H, C)).to(torch.float32)
    xq_out, xk_out = rope_32.rotate_qk(xq_32, xk_32)
    assert xq_out.dtype == torch.float32
    xq_out, xk_out = rope_64.rotate_qk(xq_32, xk_32)
    assert xq_out.dtype == torch.float32


def test_transformer_with_rope():
    set_efficient_attention_backend('torch')
    torch.manual_seed(1234)
    for pos in ['rope', 'sin_rope']:
        tr = StreamingTransformer(
            16, 4, 2, custom=True, dropout=0., layer_scale=0.1,
            positional_embedding=pos)
        tr.eval()
        steps = 12
        x = torch.randn(3, steps, 16)

        out = tr(x)
        assert list(out.shape) == list(x.shape)


@torch.no_grad()
def test_rope_streaming():
    set_efficient_attention_backend('torch')
    torch.manual_seed(1234)
    tr = StreamingTransformer(
        16, 4, 2, causal=True, dropout=0.,
        custom=True, positional_embedding='rope')
    tr.eval()
    steps = 12
    x = torch.randn(3, steps, 16)

    ref = tr(x)

    with tr.streaming():
        outs = []
        frame_sizes = [1] * steps

        for frame_size in frame_sizes:
            frame = x[:, :frame_size]
            x = x[:, frame_size:]
            outs.append(tr(frame))

    out = torch.cat(outs, dim=1)
    assert list(out.shape) == [3, steps, 16]
    delta = torch.norm(out - ref) / torch.norm(out)
    assert delta < 1e-6, delta


@torch.no_grad()
def test_rope_streaming_past_context():
    set_efficient_attention_backend('torch')
    torch.manual_seed(1234)

    for context in [None, 10]:
        tr = StreamingTransformer(
            16, 4, 1 if context else 2,
            causal=True, past_context=context, custom=True,
            dropout=0., positional_embedding='rope')
        tr.eval()

        steps = 20
        x = torch.randn(3, steps, 16)
        ref = tr(x)

        with tr.streaming():
            outs = []
            frame_sizes = [1] * steps

            for frame_size in frame_sizes:
                frame = x[:, :frame_size]
                x = x[:, frame_size:]
                outs.append(tr(frame))

        out = torch.cat(outs, dim=1)
        assert list(out.shape) == [3, steps, 16]
        delta = torch.norm(out - ref) / torch.norm(out)
        assert delta < 1e-6, delta


def test_rope_memory_efficient():
    set_efficient_attention_backend('torch')
    torch.manual_seed(1234)
    tr = StreamingTransformer(
        16, 4, 2, custom=True, dropout=0., layer_scale=0.1,
        positional_embedding='rope')
    tr_mem_efficient = StreamingTransformer(
        16, 4, 2, dropout=0., memory_efficient=True, layer_scale=0.1,
        positional_embedding='rope')
    tr_mem_efficient.load_state_dict(tr.state_dict())
    tr.eval()
    steps = 12
    x = torch.randn(3, steps, 16)

    with torch.no_grad():
        y = tr(x)
        y2 = tr_mem_efficient(x)
        # Check at float precision b/c this is the rope default.
        assert torch.allclose(y, y2, atol=1e-7), (y - y2).norm()


def test_rope_with_xpos():
    set_efficient_attention_backend('torch')
    B, T, H, C = 8, 75, 16, 128

    rope = RotaryEmbedding(dim=C, xpos=True)
    xq = torch.rand((B, T, H, C))
    xk = torch.rand((B, T, H, C))
    xq_out, xk_out = rope.rotate_qk(xq, xk, start=7)

    assert list(xq_out.shape) == [B, T, H, C]
    assert list(xk_out.shape) == [B, T, H, C]


def test_positional_scale():
    set_efficient_attention_backend('torch')
    B, T, H, C = 8, 75, 16, 128

    rope = RotaryEmbedding(dim=C, xpos=True, scale=0.0)
    xq = torch.rand((B, T, H, C))
    xk = torch.rand((B, T, H, C))
    xq_out, xk_out = rope.rotate_qk(xq, xk, start=7)

    assert torch.allclose(xq, xq_out)
    assert torch.allclose(xk, xk_out)
