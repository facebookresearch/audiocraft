# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import pytest
import torch

from audiocraft.modules.transformer import (
    StreamingMultiheadAttention, StreamingTransformer, set_efficient_attention_backend)


def test_transformer_causal_streaming():
    torch.manual_seed(1234)

    for context, custom in product([None, 10], [False, True]):
        # Test that causality and receptive fields are properly handled.
        # looking at the gradients
        tr = StreamingTransformer(
            16, 4, 1 if context else 2,
            causal=True, past_context=context, custom=custom,
            dropout=0.)
        steps = 20
        for k in [0, 10, 15, 19]:
            x = torch.randn(4, steps, 16, requires_grad=True)
            y = tr(x)
            y[:, k].abs().sum().backward()
            if k + 1 < steps:
                assert torch.allclose(x.grad[:, k + 1:], torch.tensor(0.)), x.grad[:, k + 1:].norm()
            assert not torch.allclose(x.grad[:, :k + 1], torch.tensor(0.)), x.grad[:, :k + 1].norm()
            if context is not None and k > context:
                limit = k - context - 1
                assert torch.allclose(x.grad[:, :limit],
                                      torch.tensor(0.)), x.grad[:, :limit].norm()

        # Now check that streaming gives the same result at batch eval.
        x = torch.randn(4, steps, 16)
        y = tr(x)
        ys = []
        with tr.streaming():
            for k in range(steps):
                chunk = x[:, k:k + 1, :]
                ys.append(tr(chunk))
        y_stream = torch.cat(ys, dim=1)
        delta = torch.norm(y_stream - y) / torch.norm(y)
        assert delta < 1e-6, delta


def test_transformer_vs_pytorch():
    torch.manual_seed(1234)
    # Check that in the non causal setting, we get the same result as
    # PyTorch Transformer encoder.
    for custom in [False, True]:
        tr = StreamingTransformer(
            16, 4, 2,
            causal=False, custom=custom, dropout=0., positional_scale=0.)
        layer = torch.nn.TransformerEncoderLayer(16, 4, dropout=0., batch_first=True)
        tr_ref = torch.nn.TransformerEncoder(layer, 2)
        tr.load_state_dict(tr_ref.state_dict())

        x = torch.randn(4, 20, 16)
        y = tr(x)
        y2 = tr_ref(x)
        delta = torch.norm(y2 - y) / torch.norm(y)
        assert delta < 1e-6, delta


def test_streaming_api():
    tr = StreamingTransformer(16, 4, 2, causal=True, dropout=0.)
    tr.eval()
    steps = 12
    x = torch.randn(1, steps, 16)

    with torch.no_grad():
        with tr.streaming():
            _ = tr(x[:, :1])
            state = {k: v.clone() for k, v in tr.get_streaming_state().items()}
            y = tr(x[:, 1:2])
            tr.set_streaming_state(state)
            y2 = tr(x[:, 1:2])
            assert torch.allclose(y, y2), (y - y2).norm()
            assert tr.flush() is None


def test_memory_efficient():
    for backend in ['torch']:
        torch.manual_seed(1234)
        set_efficient_attention_backend(backend)

        tr = StreamingTransformer(
            16, 4, 2, custom=True, dropout=0., layer_scale=0.1)
        tr_mem_efficient = StreamingTransformer(
            16, 4, 2, dropout=0., memory_efficient=True, layer_scale=0.1)
        tr_mem_efficient.load_state_dict(tr.state_dict())
        tr.eval()
        steps = 12
        x = torch.randn(3, steps, 16)

        with torch.no_grad():
            y = tr(x)
            y2 = tr_mem_efficient(x)
            assert torch.allclose(y, y2), ((y - y2).norm(), backend)


def test_attention_as_float32():
    torch.manual_seed(1234)
    cases = [
        {'custom': True},
        {'custom': False},
    ]
    for case in cases:
        tr = StreamingTransformer(16, 4, 2, dropout=0., dtype=torch.bfloat16, **case)
        tr_float32 = StreamingTransformer(
            16, 4, 2, dropout=0., attention_as_float32=True, dtype=torch.bfloat16, **case)
        if not case['custom']:
            # we are not using autocast here because it doesn't really
            # work as expected on CPU, so we have to manually cast the weights of the MHA.
            for layer in tr_float32.layers:
                layer.self_attn.mha.to(torch.float32)
        tr_float32.load_state_dict(tr.state_dict())
        steps = 12
        x = torch.randn(3, steps, 16, dtype=torch.bfloat16)

        with torch.no_grad():
            y = tr(x)
            y2 = tr_float32(x)
            assert not torch.allclose(y, y2), (y - y2).norm()


@torch.no_grad()
def test_streaming_memory_efficient():
    for backend in ['torch']:
        torch.manual_seed(1234)
        set_efficient_attention_backend(backend)
        tr = StreamingTransformer(16, 4, 2, causal=True, dropout=0., custom=True)
        tr_mem_efficient = StreamingTransformer(
            16, 4, 2, dropout=0., memory_efficient=True, causal=True)
        tr.load_state_dict(tr_mem_efficient.state_dict())
        tr.eval()
        tr_mem_efficient.eval()
        steps = 12
        x = torch.randn(3, steps, 16)

        ref = tr(x)

        with tr_mem_efficient.streaming():
            outs = []
            # frame_sizes = [2] + [1] * (steps - 2)
            frame_sizes = [1] * steps

            for frame_size in frame_sizes:
                frame = x[:, :frame_size]
                x = x[:, frame_size:]
                outs.append(tr_mem_efficient(frame))

        out = torch.cat(outs, dim=1)
        delta = torch.norm(out - ref) / torch.norm(out)
        assert delta < 1e-6, delta


def test_cross_attention():
    torch.manual_seed(1234)
    for norm_first in [True, False]:
        m = StreamingTransformer(
            16, 4, 2, cross_attention=False, norm_first=norm_first, dropout=0., custom=True)
        m_cross = StreamingTransformer(
            16, 4, 2, cross_attention=True, norm_first=norm_first, dropout=0., custom=True)
        m_cross.load_state_dict(m.state_dict(), strict=False)
        x = torch.randn(2, 5, 16)
        cross_x = torch.randn(2, 3, 16)
        y_ref = m(x)
        y_cross_zero = m_cross(x, cross_attention_src=0 * cross_x)
        # With norm_first, the two should be exactly the same,
        # but with norm_first=False, we get 2 normalization in a row
        # and the epsilon value leads to a tiny change.
        atol = 0. if norm_first else 1e-6
        print((y_ref - y_cross_zero).norm() / y_ref.norm())
        assert torch.allclose(y_ref, y_cross_zero, atol=atol)

        # We now expect a difference even with a generous atol of 1e-2.
        y_cross = m_cross(x, cross_attention_src=cross_x)
        assert not torch.allclose(y_cross, y_cross_zero, atol=1e-2)

        with pytest.raises(AssertionError):
            _ = m_cross(x)
            _ = m(x, cross_attention_src=cross_x)


def test_cross_attention_compat():
    torch.manual_seed(1234)
    num_heads = 2
    dim = num_heads * 64
    with pytest.raises(AssertionError):
        StreamingMultiheadAttention(dim, num_heads, causal=True, cross_attention=True)

    cross_attn = StreamingMultiheadAttention(
        dim, num_heads, dropout=0, cross_attention=True, custom=True)
    ref_attn = torch.nn.MultiheadAttention(dim, num_heads, dropout=0, batch_first=True)

    # We can load the regular attention state dict
    # so we have compat when loading old checkpoints.
    cross_attn.load_state_dict(ref_attn.state_dict())

    queries = torch.randn(3, 7, dim)
    keys = torch.randn(3, 9, dim)
    values = torch.randn(3, 9, dim)

    y = cross_attn(queries, keys, values)[0]
    y_ref = ref_attn(queries, keys, values)[0]
    assert torch.allclose(y, y_ref, atol=1e-7), (y - y_ref).norm() / y_ref.norm()

    # Now let's check that streaming is working properly.
    with cross_attn.streaming():
        ys = []
        for step in range(queries.shape[1]):
            ys.append(cross_attn(queries[:, step: step + 1], keys, values)[0])
    y_streaming = torch.cat(ys, dim=1)
    assert torch.allclose(y_streaming, y, atol=1e-7)


def test_repeat_kv():
    torch.manual_seed(1234)
    num_heads = 8
    kv_repeat = 4
    dim = num_heads * 64
    with pytest.raises(AssertionError):
        mha = StreamingMultiheadAttention(
            dim, num_heads, causal=True, kv_repeat=kv_repeat, cross_attention=True)
        mha = StreamingMultiheadAttention(
            dim, num_heads, causal=True, kv_repeat=kv_repeat)
    mha = StreamingMultiheadAttention(
        dim, num_heads, causal=True, kv_repeat=kv_repeat, custom=True)
    x = torch.randn(4, 18, dim)
    y = mha(x, x, x)[0]
    assert x.shape == y.shape


def test_qk_layer_norm():
    torch.manual_seed(1234)
    tr = StreamingTransformer(
        16, 4, 2, custom=True, dropout=0., qk_layer_norm=True, bias_attn=False)
    steps = 12
    x = torch.randn(3, steps, 16)
    y = tr(x)

    tr = StreamingTransformer(
        16, 4, 2, custom=True, dropout=0., qk_layer_norm=True, cross_attention=True)
    z = torch.randn(3, 21, 16)
    y = tr(x, cross_attention_src=z)
    assert y.shape == x.shape
