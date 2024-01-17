# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transformer model, with streaming support, xformer attention support
and easy causal attention with a potentially finite receptive field.

See `StreamingTransformer` for more information.

Unlike regular PyTorch Transformer, we make the hard choice that batches are first.
"""

import typing as tp

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from xformers import ops

from .rope import RotaryEmbedding
from .streaming import StreamingModule

_efficient_attention_backend: str = 'torch'


def set_efficient_attention_backend(backend: str = 'torch'):
    # Using torch by default, it seems a bit faster on older P100 GPUs (~20% faster).
    global _efficient_attention_backend
    assert _efficient_attention_backend in ['xformers', 'torch']
    _efficient_attention_backend = backend


def _get_attention_time_dimension(memory_efficient: bool) -> int:
    if _efficient_attention_backend == 'torch' and memory_efficient:
        return 2
    else:
        return 1


def _is_profiled() -> bool:
    # Return true if we are currently running with a xformers profiler activated.
    try:
        from xformers.profiler import profiler
    except ImportError:
        return False
    return profiler._Profiler._CURRENT_PROFILER is not None


def create_norm_fn(norm_type: str, dim: int, **kwargs) -> nn.Module:
    """Create normalization module for transformer encoder layer.

    Args:
        norm_type (str): Normalization method.
        dim (int): Dimension of the normalized layer.
        **kwargs (dict): Additional parameters for normalization layer.
    Returns:
        nn.Module: Normalization module.
    """
    if norm_type == 'layer_norm':
        return nn.LayerNorm(dim, eps=1e-5, **kwargs)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


def create_sin_embedding(positions: torch.Tensor, dim: int, max_period: float = 10000,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create sinusoidal positional embedding, with shape `[B, T, C]`.

    Args:
        positions (torch.Tensor): LongTensor of positions.
        dim (int): Dimension of the embedding.
        max_period (float): Maximum period of the cosine/sine functions.
        dtype (torch.dtype or str): dtype to use to generate the embedding.
    Returns:
        torch.Tensor: Sinusoidal positional embedding.
    """
    # We aim for BTC format
    assert dim % 2 == 0
    half_dim = dim // 2
    positions = positions.to(dtype)
    adim = torch.arange(half_dim, device=positions.device, dtype=dtype).view(1, 1, -1)
    max_period_tensor = torch.full([], max_period, device=positions.device, dtype=dtype)  # avoid sync point
    phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


def expand_repeated_kv(x: torch.Tensor, n_rep: int, memory_efficient: bool) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep) from xlformers."""
    if n_rep == 1:
        return x
    if _efficient_attention_backend == 'torch' and memory_efficient:
        bs, n_kv_heads, slen, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(bs, n_kv_heads, n_rep, slen, head_dim)
            .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
        )
    else:
        bs, slen, n_kv_heads, head_dim = x.shape
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )


class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonally the residual outputs close to 0, with a learnt scale.

    Args:
        channels (int): Number of channels.
        init (float): Initial scale.
        channel_last (bool): If True, expect `[*, C]` shaped tensors, otherwise, `[*, C, T]`.
        device (torch.device or str, optional): Device on which to initialize the module.
        dtype (torch.dtype, optional): dtype to use to initialize the module.
    """
    def __init__(self, channels: int, init: float = 1e-4, channel_last: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(
            torch.full((channels,), init,
                       requires_grad=True, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x


class StreamingMultiheadAttention(StreamingModule):
    """Similar to `nn.MultiheadAttention` but with support for streaming, causal evaluation.

    Args:
        embed_dim (int): Dimension to project to.
        num_heads (int): Number of heads.
        dropout (float): Dropout level.
        bias (bool): Use bias in projections.
        causal (bool): Causal mask applied automatically.
        past_context (int, optional): Receptive field for the causal mask, infinite if None.
        custom (bool): Use custom MHA implementation, for testing / benchmarking.
        memory_efficient (bool): Use xformers based memory efficient attention.
        attention_as_float32 (bool): Perform the attention as float32
            (especially important with memory_efficient as autocast won't do this automatically).
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        cross_attention: Should be true when used as a cross attention.
            All keys and values must be available at once, streaming is only for the queries.
            Cannot be used with `causal` or `rope` (as it wouldn't make sens to
            interpret the time steps in the keys relative to those in the queries).
        safe_streaming (bool): Bug fix, will go away with xformers update.
        qk_layer_norm (bool): Layer normalization applied to queries and keys before dot product.
        kv_repeat (int): If > 1, will repeat keys and queries multiple times (need to divide num_heads).
            This will lead to faster decoding time on A100 or other GPUs with tensorcore.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True,
                 causal: bool = False, past_context: tp.Optional[int] = None, custom: bool = False,
                 memory_efficient: bool = False, attention_as_float32: bool = False,
                 rope: tp.Optional[RotaryEmbedding] = None, cross_attention: bool = False,
                 safe_streaming: bool = True, qk_layer_norm: bool = False, kv_repeat: int = 1,
                 device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        if past_context is not None:
            assert causal

        self.embed_dim = embed_dim
        self.causal = causal
        self.past_context = past_context
        self.memory_efficient = memory_efficient
        self.attention_as_float32 = attention_as_float32
        self.rope = rope
        self.cross_attention = cross_attention
        self.safe_streaming = safe_streaming
        self.num_heads = num_heads
        self.dropout = dropout
        self.kv_repeat = kv_repeat
        if cross_attention:
            assert not causal, "Causal cannot work with cross attention."
            assert rope is None, "Rope cannot work with cross attention."

        if memory_efficient:
            _verify_xformers_memory_efficient_compat()

        self.custom = _is_custom(custom, memory_efficient)
        if self.custom:
            out_dim = embed_dim
            assert num_heads % kv_repeat == 0
            assert not cross_attention or kv_repeat == 1
            num_kv = num_heads // kv_repeat
            kv_dim = (embed_dim // num_heads) * num_kv
            out_dim += 2 * kv_dim
            in_proj = nn.Linear(embed_dim, out_dim, bias=bias, **factory_kwargs)
            # We try to follow the default PyTorch MHA convention, to easily compare results.
            self.in_proj_weight = in_proj.weight
            self.in_proj_bias = in_proj.bias
            if bias:
                self.in_proj_bias.data.zero_()  # Following Pytorch convention
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
            if bias:
                self.out_proj.bias.data.zero_()
        else:
            assert not qk_layer_norm
            assert kv_repeat == 1
            self.mha = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, bias=bias, batch_first=True,
                **factory_kwargs)
        self.qk_layer_norm = qk_layer_norm
        if qk_layer_norm:
            assert self.custom
            assert kv_repeat == 1
            ln_dim = embed_dim
            self.q_layer_norm = nn.LayerNorm(ln_dim)
            self.k_layer_norm = nn.LayerNorm(ln_dim)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if not self.custom:
            # Support compat with regular MHA
            keys = [n for n, _ in self.mha.named_parameters()]
            for key in keys:
                if prefix + key in state_dict:
                    state_dict[prefix + "mha." + key] = state_dict.pop(prefix + key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _get_mask(self, current_steps: int, device: torch.device, dtype: torch.dtype):
        # Return a causal mask, accounting for potentially stored past keys/values
        # We actually return a bias for the attention score, as this has the same
        # convention both in the builtin MHA in Pytorch, and Xformers functions.
        time_dim = _get_attention_time_dimension(self.memory_efficient)
        if self.memory_efficient:
            from xformers.ops import LowerTriangularMask
            if current_steps == 1:
                # If we only have one step, then we do not need a mask.
                return None
            elif 'past_keys' in self._streaming_state:
                raise RuntimeError("Not supported at the moment")
            else:
                # Then we can safely use a lower triangular mask
                return LowerTriangularMask()
        if self._streaming_state:
            past_keys = self._streaming_state['past_keys']
            past_steps = past_keys.shape[time_dim]
        else:
            past_steps = 0

        queries_pos = torch.arange(
            past_steps, current_steps + past_steps, device=device).view(-1, 1)
        keys_pos = torch.arange(past_steps + current_steps, device=device).view(1, -1)
        delta = queries_pos - keys_pos
        valid = delta >= 0
        if self.past_context is not None:
            valid &= (delta <= self.past_context)
        return torch.where(
            valid,
            torch.zeros([], device=device, dtype=dtype),
            torch.full([], float('-inf'), device=device, dtype=dtype))

    def _complete_kv(self, k, v):
        time_dim = _get_attention_time_dimension(self.memory_efficient)
        if self.cross_attention:
            # With cross attention we assume all keys and values
            # are already available, and streaming is with respect
            # to the queries only.
            return k, v
        # Complete the key/value pair using the streaming state.
        if self._streaming_state:
            pk = self._streaming_state['past_keys']
            nk = torch.cat([pk, k], dim=time_dim)
            if v is k:
                nv = nk
            else:
                pv = self._streaming_state['past_values']
                nv = torch.cat([pv, v], dim=time_dim)
        else:
            nk = k
            nv = v

        assert nk.shape[time_dim] == nv.shape[time_dim]
        offset = 0
        if self.past_context is not None:
            offset = max(0, nk.shape[time_dim] - self.past_context)
        if self._is_streaming:
            self._streaming_state['past_keys'] = nk[:, offset:]
            if v is not k:
                self._streaming_state['past_values'] = nv[:, offset:]
            if 'offset' in self._streaming_state:
                self._streaming_state['offset'] += offset
            else:
                self._streaming_state['offset'] = torch.tensor(0)
        return nk, nv

    def _apply_rope(self, query: torch.Tensor, key: torch.Tensor):
        time_dim = _get_attention_time_dimension(self.memory_efficient)
        # Apply rope embeddings to query and key tensors.
        assert self.rope is not None
        if 'past_keys' in self._streaming_state:
            past_keys_offset = self._streaming_state['past_keys'].shape[1]
        else:
            past_keys_offset = 0
        if 'offset' in self._streaming_state:
            past_context_offset = int(self._streaming_state['offset'].item())
        else:
            past_context_offset = 0
        streaming_offset = past_context_offset + past_keys_offset
        return self.rope.rotate_qk(query, key, start=streaming_offset, time_dim=time_dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask=None, need_weights=False, attn_mask=None,
                average_attn_weights=True, is_causal=False):
        assert not is_causal, ("New param added in torch 2.0.1 not supported, "
                               "use the causal args in the constructor.")

        time_dim = _get_attention_time_dimension(self.memory_efficient)
        if time_dim == 2:
            layout = "b h t d"
        else:
            layout = "b t h d"
        dtype = query.dtype
        if self._is_streaming:
            assert self.causal or self.cross_attention, \
                "Streaming only available for causal or cross attention"

        custom_attn_mask = attn_mask is not None

        if self.causal:
            assert attn_mask is None
            # At the moment we specialize only for the self-attention case.
            assert query.shape[1] == key.shape[1], "Causal only for same length query / key / value"
            assert value.shape[1] == key.shape[1], "Causal only for same length query / key / value"
            attn_mask = self._get_mask(query.shape[1], query.device, query.dtype)

        if self.custom:
            # custom implementation
            assert need_weights is False
            assert key_padding_mask is None
            if self.cross_attention:
                # Different queries, keys, values, we have to spit manually the weights
                # before applying the linear.
                dim = self.in_proj_weight.shape[0] // 3
                if self.in_proj_bias is None:
                    bias_q, bias_k, bias_v = None, None, None
                else:
                    bias_q = self.in_proj_bias[:dim]
                    bias_k = self.in_proj_bias[dim: 2 * dim]
                    bias_v = self.in_proj_bias[2 * dim:]
                q = nn.functional.linear(query, self.in_proj_weight[:dim], bias_q)
                # todo: when streaming, we could actually save k, v and check the shape actually match.
                k = nn.functional.linear(key, self.in_proj_weight[dim: 2 * dim], bias_k)
                v = nn.functional.linear(value, self.in_proj_weight[2 * dim:], bias_v)
                if self.qk_layer_norm is True:
                    q = self.q_layer_norm(q)
                    k = self.k_layer_norm(k)
                q, k, v = [rearrange(x, f"b t (h d) -> {layout}", h=self.num_heads) for x in [q, k, v]]
            else:
                if not _is_profiled():
                    # profiling breaks that propertysomehow.
                    assert query is key, "specialized implementation"
                    assert value is key, "specialized implementation"
                projected = nn.functional.linear(query, self.in_proj_weight, self.in_proj_bias)
                if self.kv_repeat == 1:
                    if time_dim == 2:
                        bound_layout = "b h p t d"
                    else:
                        bound_layout = "b t p h d"
                    packed = rearrange(projected, f"b t (p h d) -> {bound_layout}", p=3, h=self.num_heads)
                    q, k, v = ops.unbind(packed, dim=2)
                else:
                    embed_dim = self.embed_dim
                    per_head_dim = (embed_dim // self.num_heads)
                    kv_heads = self.num_heads // self.kv_repeat
                    q = projected[:, :, :embed_dim]
                    start = embed_dim
                    end = start + per_head_dim * kv_heads
                    k = projected[:, :, start: end]
                    v = projected[:, :, end:]
                    q = rearrange(q, f"b t (h d) -> {layout}", h=self.num_heads)
                    k = rearrange(k, f"b t (h d) -> {layout}", h=kv_heads)
                    v = rearrange(v, f"b t (h d) -> {layout}", h=kv_heads)

                if self.qk_layer_norm is True:
                    assert self.kv_repeat == 1
                    q, k = [rearrange(x, f"{layout} -> b t (h d)") for x in [q, k]]
                    q = self.q_layer_norm(q)
                    k = self.k_layer_norm(k)
                    q, k = [rearrange(x, f"b t (h d) -> {layout}", h=self.num_heads) for x in [q, k]]
                if self.rope:
                    q, k = self._apply_rope(q, k)
                k, v = self._complete_kv(k, v)
                if self.kv_repeat > 1:
                    k = expand_repeated_kv(k, self.kv_repeat, self.memory_efficient)
                    v = expand_repeated_kv(v, self.kv_repeat, self.memory_efficient)
            if self.attention_as_float32:
                q, k, v = [x.float() for x in [q, k, v]]
            if self.memory_efficient:
                if custom_attn_mask:
                    # When using a custom attn mask:
                    # Move to query's device, repeat for each sample, remove align8 padding
                    seq_len = query.shape[1]
                    attn_mask = attn_mask.to(q.dtype)
                    attn_mask = attn_mask.repeat((q.shape[0], 1, 1, 1))
                    attn_mask = attn_mask[..., :seq_len, :seq_len]

                p = self.dropout if self.training else 0
                if _efficient_attention_backend == 'torch':
                    x = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, is_causal=attn_mask is not None, dropout_p=p)
                else:
                    x = ops.memory_efficient_attention(q, k, v, attn_mask, p=p)
            else:
                # We include the dot product as float32, for consistency
                # with the other implementations that include that step
                # as part of the attention. Note that when using `autocast`,
                # the einsums would be done as bfloat16, but the softmax
                # would be done as bfloat16, so `attention_as_float32` will
                # extend a bit the range of operations done in float32,
                # although this should make no difference.
                q = q / q.shape[-1] ** 0.5
                key_layout = layout.replace('t', 'k')
                query_layout = layout
                if self._is_streaming and self.safe_streaming and q.device.type == 'cuda':
                    with torch.autocast(device_type=q.device.type, dtype=torch.float32):
                        pre_w = torch.einsum(f"{query_layout},{key_layout}-> b h t k", q, k)
                else:
                    pre_w = torch.einsum(f"{query_layout},{key_layout}-> b h t k", q, k)
                if attn_mask is not None:
                    pre_w = pre_w + attn_mask
                w = torch.softmax(pre_w, dim=-1)
                w = F.dropout(w, self.dropout, training=self.training).to(v)
                # Key and value have the same format.
                x = torch.einsum(f"b h t k, {key_layout} -> {layout}", w, v)
            x = x.to(dtype)
            x = rearrange(x, f"{layout} -> b t (h d)", h=self.num_heads)
            x = self.out_proj(x)
        else:
            key, value = self._complete_kv(key, value)
            if self.attention_as_float32:
                query, key, value = [x.float() for x in [query, key, value]]
            x, _ = self.mha(
                query, key, value, key_padding_mask,
                need_weights, attn_mask, average_attn_weights)
            x = x.to(dtype)

        return x, None


class StreamingTransformerLayer(nn.TransformerEncoderLayer):
    """TransformerLayer with Streaming / Causal support.
    This also integrates cross_attention, when passing `cross_attention=True`,
    rather than having two separate classes like in PyTorch.

    Args:
        d_model (int): Dimension of the data.
        num_heads (int): Number of heads.
        dim_feedforward (int): Intermediate dimension of FF module.
        dropout (float): Dropout both for MHA and FF.
        bias_ff (bool): Use bias for FF.
        bias_attn (bool): Use bias for MHA.
        causal (bool): Causal mask applied automatically.
        past_context (int, optional): Receptive field for the causal mask, infinite if None.
        custom (bool): Use custom MHA implementation, for testing / benchmarking.
        memory_efficient (bool): Use xformers based memory efficient attention.
        attention_as_float32 (bool): Perform the attention as float32
            (especially important with memory_efficient as autocast won't do this automatically).
        qk_layer_norm (bool): Layer normalization applied to queries and keys before dot product in attention.
        qk_layer_norm_cross (bool): Same for the cross attention.
        cross_attention (bool): If True, expect to get secondary input for cross-attention.
            Cross attention will use the default MHA, as it typically won't require
            special treatment.
        layer_scale (float, optional): If not None, LayerScale will be used with
            the given value as initial scale.
        rope (`RotaryEmbedding`, optional): Rope embedding to use.
        attention_dropout (float, optional): If not None, separate the value of the dimension dropout
            in FFN and of the attention dropout.
        kv_repeat (int): If > 1, will repeat keys and queries multiple times (need to divide num_heads).
            This will lead to faster decoding time on A100 or other GPUs with tensorcore.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
        **kwargs: See `nn.TransformerEncoderLayer`.
    """
    def __init__(self, d_model: int, num_heads: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 bias_ff: bool = True, bias_attn: bool = True, causal: bool = False,
                 past_context: tp.Optional[int] = None, custom: bool = False,
                 memory_efficient: bool = False, attention_as_float32: bool = False,
                 qk_layer_norm: bool = False, qk_layer_norm_cross: bool = False,
                 cross_attention: bool = False, layer_scale: tp.Optional[float] = None,
                 rope: tp.Optional[RotaryEmbedding] = None, attention_dropout: tp.Optional[float] = None,
                 kv_repeat: int = 1, norm: str = 'layer_norm', device=None, dtype=None, **kwargs):
        super().__init__(d_model, num_heads, dim_feedforward, dropout,
                         device=device, dtype=dtype, batch_first=True, **kwargs)
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Redefine self_attn to our streaming multi-head attention
        attn_kwargs: tp.Dict[str, tp.Any] = {
            'embed_dim': d_model,
            'num_heads': num_heads,
            'dropout': dropout if attention_dropout is None else attention_dropout,
            'bias': bias_attn,
            'custom': custom,
            'memory_efficient': memory_efficient,
            'attention_as_float32': attention_as_float32,
        }
        self.self_attn: StreamingMultiheadAttention = StreamingMultiheadAttention(
            causal=causal, past_context=past_context, rope=rope, qk_layer_norm=qk_layer_norm,
            kv_repeat=kv_repeat, **attn_kwargs, **factory_kwargs)  # type: ignore
        # Redefine feedforward layers to expose bias parameter
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias_ff, **factory_kwargs)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias_ff, **factory_kwargs)

        self.layer_scale_1: nn.Module
        self.layer_scale_2: nn.Module
        if layer_scale is None:
            self.layer_scale_1 = nn.Identity()
            self.layer_scale_2 = nn.Identity()
        else:
            self.layer_scale_1 = LayerScale(d_model, layer_scale, **factory_kwargs)
            self.layer_scale_2 = LayerScale(d_model, layer_scale, **factory_kwargs)

        self.cross_attention: tp.Optional[nn.Module] = None
        if cross_attention:
            self.cross_attention = StreamingMultiheadAttention(
                cross_attention=True, qk_layer_norm=qk_layer_norm_cross,
                **attn_kwargs, **factory_kwargs)
            # Norm and dropout
            self.dropout_cross = nn.Dropout(dropout)
            # eps value matching that used in PyTorch reference implementation.
            self.norm_cross = nn.LayerNorm(d_model, eps=1e-5, **factory_kwargs)
            self.layer_scale_cross: nn.Module
            if layer_scale is None:
                self.layer_scale_cross = nn.Identity()
            else:
                self.layer_scale_cross = LayerScale(d_model, layer_scale, **factory_kwargs)
        self.norm1 = create_norm_fn(norm, d_model, **factory_kwargs)  # type: ignore
        self.norm2 = create_norm_fn(norm, d_model, **factory_kwargs)  # type: ignore

    def _cross_attention_block(self, src: torch.Tensor,
                               cross_attention_src: torch.Tensor) -> torch.Tensor:
        assert self.cross_attention is not None
        # queries are from src, keys and values from cross_attention_src.
        x = self.cross_attention(
            src, cross_attention_src, cross_attention_src, need_weights=False)[0]
        return self.dropout_cross(x)  # type: ignore

    def forward(self, src: torch.Tensor, src_mask: tp.Optional[torch.Tensor] = None,  # type: ignore
                src_key_padding_mask: tp.Optional[torch.Tensor] = None,
                cross_attention_src: tp.Optional[torch.Tensor] = None):
        if self.cross_attention is None:
            assert cross_attention_src is None
        else:
            assert cross_attention_src is not None
        x = src
        if self.norm_first:
            x = x + self.layer_scale_1(
                self._sa_block(self.norm1(x), src_mask, src_key_padding_mask))
            if cross_attention_src is not None:
                x = x + self.layer_scale_cross(
                    self._cross_attention_block(
                        self.norm_cross(x), cross_attention_src))
            x = x + self.layer_scale_2(self._ff_block(self.norm2(x)))
        else:
            x = self.norm1(x + self.layer_scale_1(
                self._sa_block(x, src_mask, src_key_padding_mask)))
            if cross_attention_src is not None:
                x = self.norm_cross(
                    x + self.layer_scale_cross(
                        self._cross_attention_block(src, cross_attention_src)))
            x = self.norm2(x + self.layer_scale_2(self._ff_block(x)))
        return x


class StreamingTransformer(StreamingModule):
    """Transformer with Streaming / Causal support.

    Args:
        d_model (int): Dimension of the data.
        num_heads (int): Number of heads.
        dim_feedforward (int): Intermediate dimension of FF module.
        dropout (float): Dropout both for MHA and FF.
        bias_ff (bool): Use bias for FF.
        bias_attn (bool): Use bias for MHA.
        causal (bool): Causal mask applied automatically.
        past_context (int, optional): Receptive field for the causal mask, infinite if None.
        custom (bool): Use custom MHA implementation, for testing / benchmarking.
        memory_efficient (bool): Use xformers based memory efficient attention.
        attention_as_float32 (bool): Perform the attention as float32
            (especially important with memory_efficient as autocast won't do this automatically).
        cross_attention (bool): If True, expect to get secondary input for cross-attention.
        layer_scale (float, optional): If not None, LayerScale will be used
            with the given value as initial scale.
        positional_embedding (str): Positional embedding strategy (sin, rope, or sin_rope).
        max_period (float): Maximum period of the time embedding.
        positional_scale (float): Scale of positional embedding, set to 0 to deactivate.
        xpos (bool): Apply xpos exponential decay to positional embedding (rope only).
        lr (float, optional): learning rate override through the `make_optim_group` API.
        weight_decay (float, optional): Weight_decay override through the `make_optim_group` API.
        layer_class: (subclass of `StreamingTransformerLayer): class to use
            to initialize the layers, allowing further customization outside of AudioCraft.
        checkpointing (str): Checkpointing strategy to reduce memory usage.
            No checkpointing if set to 'none'. Per layer checkpointing using PyTorch
            if set to 'torch' (entire layer checkpointed, i.e. linears are evaluated twice,
            minimal memory usage, but maximal runtime). Finally, `xformers_default` provide
            a policy for opting-out some operations of the checkpointing like
            linear layers and attention, providing a middle ground between speed and memory.
        device (torch.device, optional): Device on which to initialize.
        dtype (torch.dtype, optional): dtype to use.
        **kwargs: See `nn.TransformerEncoderLayer`.
    """
    def __init__(self, d_model: int, num_heads: int, num_layers: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, bias_ff: bool = True, bias_attn: bool = True,
                 causal: bool = False, past_context: tp.Optional[int] = None,
                 custom: bool = False, memory_efficient: bool = False, attention_as_float32: bool = False,
                 cross_attention: bool = False, layer_scale: tp.Optional[float] = None,
                 positional_embedding: str = 'sin', max_period: float = 10_000, positional_scale: float = 1.,
                 xpos: bool = False, lr: tp.Optional[float] = None, weight_decay: tp.Optional[float] = None,
                 layer_class: tp.Type[StreamingTransformerLayer] = StreamingTransformerLayer,
                 checkpointing: str = 'none', device=None, dtype=None, **kwargs):
        super().__init__()
        assert d_model % num_heads == 0

        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.positional_scale = positional_scale
        self.weight_decay = weight_decay
        self.lr = lr

        assert positional_embedding in ['sin', 'rope', 'sin_rope']
        self.rope: tp.Optional[RotaryEmbedding] = None
        if self.positional_embedding in ['rope', 'sin_rope']:
            assert _is_custom(custom, memory_efficient)
            self.rope = RotaryEmbedding(d_model // num_heads, max_period=max_period,
                                        xpos=xpos, scale=positional_scale, device=device)

        self.checkpointing = checkpointing

        assert checkpointing in ['none', 'torch', 'xformers_default', 'xformers_mm']
        if self.checkpointing.startswith('xformers'):
            _verify_xformers_internal_compat()

        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            self.layers.append(
                layer_class(
                    d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward,
                    dropout=dropout, bias_ff=bias_ff, bias_attn=bias_attn,
                    causal=causal, past_context=past_context, custom=custom,
                    memory_efficient=memory_efficient, attention_as_float32=attention_as_float32,
                    cross_attention=cross_attention, layer_scale=layer_scale, rope=self.rope,
                    device=device, dtype=dtype, **kwargs))

        if self.checkpointing != 'none':
            for layer in self.layers:
                # see audiocraft/optim/fsdp.py, magic signal to indicate this requires fixing the
                # backward hook inside of FSDP...
                layer._magma_checkpointed = True  # type: ignore

    def _apply_layer(self, layer, *args, **kwargs):
        method = self.checkpointing
        if method == 'none':
            return layer(*args, **kwargs)
        elif method == 'torch':
            return torch_checkpoint(layer, *args, use_reentrant=False, **kwargs)
        elif method.startswith('xformers'):
            from xformers.checkpoint_fairinternal import checkpoint, _get_default_policy
            if method == 'xformers_default':
                # those operations will be saved, and not recomputed.
                # According to Francisco we can get smarter policies but this is a good start.
                allow_list = [
                    "xformers.efficient_attention_forward_cutlass.default",
                    "xformers_flash.flash_fwd.default",
                    "aten.addmm.default",
                    "aten.mm.default",
                ]
            elif method == 'xformers_mm':
                # those operations will be saved, and not recomputed.
                # According to Francisco we can get smarter policies but this is a good start.
                allow_list = [
                    "aten.addmm.default",
                    "aten.mm.default",
                ]
            else:
                raise ValueError(f"xformers checkpointing xformers policy {method} is not known.")
            policy_fn = _get_default_policy(allow_list)
            return checkpoint(layer, *args, policy_fn=policy_fn, **kwargs)
        else:
            raise ValueError(f"Checkpointing method {method} is unknown.")

    def forward(self, x: torch.Tensor, *args, **kwargs):
        B, T, C = x.shape

        if 'offsets' in self._streaming_state:
            offsets = self._streaming_state['offsets']
        else:
            offsets = torch.zeros(B, dtype=torch.long, device=x.device)

        if self.positional_embedding in ['sin', 'sin_rope']:
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            positions = positions + offsets.view(-1, 1, 1)
            pos_emb = create_sin_embedding(positions, C, max_period=self.max_period, dtype=x.dtype)
            x = x + self.positional_scale * pos_emb

        for layer in self.layers:
            x = self._apply_layer(layer, x, *args, **kwargs)

        if self._is_streaming:
            self._streaming_state['offsets'] = offsets + T

        return x

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        if self.weight_decay is not None:
            group["weight_decay"] = self.weight_decay
        return group


# special attention related function

def _verify_xformers_memory_efficient_compat():
    try:
        from xformers.ops import memory_efficient_attention, LowerTriangularMask  # noqa
    except ImportError:
        raise ImportError(
            "xformers is not installed. Please install it and try again.\n"
            "To install on AWS and Azure, run \n"
            "FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST='8.0'\\\n"
            "pip install -U git+https://git@github.com/fairinternal/xformers.git#egg=xformers\n"
            "To install on FAIR Cluster, run \n"
            "FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST='6.0;7.0'\\\n"
            "pip install -U git+https://git@github.com/fairinternal/xformers.git#egg=xformers\n")


def _verify_xformers_internal_compat():
    try:
        from xformers.checkpoint_fairinternal import checkpoint, _get_default_policy  # noqa
    except ImportError:
        raise ImportError(
            "Francisco's fairinternal xformers is not installed. Please install it and try again.\n"
            "To install on AWS and Azure, run \n"
            "FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST='8.0'\\\n"
            "pip install -U git+https://git@github.com/fairinternal/xformers.git#egg=xformers\n"
            "To install on FAIR Cluster, run \n"
            "FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST='6.0;7.0'\\\n"
            "pip install -U git+https://git@github.com/fairinternal/xformers.git#egg=xformers\n")


def _is_custom(custom: bool, memory_efficient: bool):
    return custom or memory_efficient
