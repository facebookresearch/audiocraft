# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
import typing as tp

from einops import rearrange
import torch
from torch import nn

from .. import quantization as qt


class CompressionModel(ABC, nn.Module):

    @abstractmethod
    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """See `EncodecModel.encode`"""
        ...

    @abstractmethod
    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """See `EncodecModel.decode`"""
        ...

    @property
    @abstractmethod
    def channels(self) -> int:
        ...

    @property
    @abstractmethod
    def frame_rate(self) -> int:
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @property
    @abstractmethod
    def cardinality(self) -> int:
        ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int:
        ...

    @property
    @abstractmethod
    def total_codebooks(self) -> int:
        ...

    @abstractmethod
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer.
        """
        ...


class EncodecModel(CompressionModel):
    """Encodec model operating on the raw waveform.

    Args:
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        quantizer (qt.BaseQuantizer): Quantizer network.
        frame_rate (int): Frame rate for the latent representation.
        sample_rate (int): Audio sample rate.
        channels (int): Number of audio channels.
        causal (bool): Whether to use a causal version of the model.
        renormalize (bool): Whether to renormalize the audio before running the model.
    """
    # we need assignement to override the property in the abstract class,
    # I couldn't find a better way...
    frame_rate: int = 0
    sample_rate: int = 0
    channels: int = 0

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 quantizer: qt.BaseQuantizer,
                 frame_rate: int,
                 sample_rate: int,
                 channels: int,
                 causal: bool = False,
                 renormalize: bool = False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.renormalize = renormalize
        self.causal = causal
        if self.causal:
            # we force disabling here to avoid handling linear overlap of segments
            # as supported in original EnCodec codebase.
            assert not self.renormalize, 'Causal model does not support renormalize'

    @property
    def total_codebooks(self):
        """Total number of quantizer codebooks available.
        """
        return self.quantizer.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer.
        """
        return self.quantizer.num_codebooks

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer.
        """
        self.quantizer.set_num_codebooks(n)

    @property
    def cardinality(self):
        """Cardinality of each codebook.
        """
        return self.quantizer.bins

    def preprocess(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        scale: tp.Optional[torch.Tensor]
        if self.renormalize:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
            scale = scale.view(-1, 1)
        else:
            scale = None
        return x, scale

    def postprocess(self,
                    x: torch.Tensor,
                    scale: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        if scale is not None:
            assert self.renormalize
            x = x * scale.view(-1, 1, 1)
        return x

    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        assert x.dim() == 3
        length = x.shape[-1]
        x, scale = self.preprocess(x)

        emb = self.encoder(x)
        q_res = self.quantizer(emb, self.frame_rate)
        out = self.decoder(q_res.x)

        # remove extra padding added by the encoder and decoder
        assert out.shape[-1] >= length, (out.shape[-1], length)
        out = out[..., :length]

        q_res.x = self.postprocess(out, scale)

        return q_res

    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        """Encode the given input tensor to quantized representation along with scale parameter.

        Args:
            x (torch.Tensor): Float tensor of shape [B, C, T]

        Returns:
            codes, scale (tp.Tuple[torch.Tensor, torch.Tensor]): Tuple composed of:
                codes a float tensor of shape [B, K, T] with K the number of codebooks used and T the timestep.
                scale a float tensor containing the scale for audio renormalizealization.
        """
        assert x.dim() == 3
        x, scale = self.preprocess(x)
        emb = self.encoder(x)
        codes = self.quantizer.encode(emb)
        return codes, scale

    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        """Decode the given codes to a reconstructed representation, using the scale to perform
        audio denormalization if needed.

        Args:
            codes (torch.Tensor): Int tensor of shape [B, K, T]
            scale (tp.Optional[torch.Tensor]): Float tensor containing the scale value.

        Returns:
            out (torch.Tensor): Float tensor of shape [B, C, T], the reconstructed audio.
        """
        emb = self.quantizer.decode(codes)
        out = self.decoder(emb)
        out = self.postprocess(out, scale)
        # out contains extra padding added by the encoder and decoder
        return out


class FlattenedCompressionModel(CompressionModel):
    """Wraps a CompressionModel and flatten its codebooks, e.g.
    instead of returning [B, K, T], return [B, S, T * (K // S)] with
    S the number of codebooks per step, and `K // S` the number of 'virtual steps'
    for each real time step.

    Args:
        model (CompressionModel): compression model to wrap.
        codebooks_per_step (int): number of codebooks to keep per step,
            this must divide the number of codebooks provided by the wrapped model.
        extend_cardinality (bool): if True, and for instance if codebooks_per_step = 1,
            if each codebook has a cardinality N, then the first codebook will
            use the range [0, N - 1], and the second [N, 2 N - 1] etc.
            On decoding, this can lead to potentially invalid sequences.
            Any invalid entry will be silently remapped to the proper range
            with a modulo.
    """
    def __init__(self, model: CompressionModel, codebooks_per_step: int = 1,
                 extend_cardinality: bool = True):
        super().__init__()
        self.model = model
        self.codebooks_per_step = codebooks_per_step
        self.extend_cardinality = extend_cardinality

    @property
    def total_codebooks(self):
        return self.model.total_codebooks

    @property
    def num_codebooks(self):
        """Active number of codebooks used by the quantizer.

        ..Warning:: this reports the number of codebooks after the flattening
        of the codebooks!
        """
        assert self.model.num_codebooks % self.codebooks_per_step == 0
        return self.codebooks_per_step

    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer.

        ..Warning:: this sets the number of codebooks **before** the flattening
        of the codebooks.
        """
        assert n % self.codebooks_per_step == 0
        self.model.set_num_codebooks(n)

    @property
    def num_virtual_steps(self) -> int:
        """Return the number of virtual steps, e.g. one real step
        will be split into that many steps.
        """
        return self.model.num_codebooks // self.codebooks_per_step

    @property
    def frame_rate(self) -> int:
        return self.model.frame_rate * self.num_virtual_steps

    @property
    def sample_rate(self) -> int:
        return self.model.sample_rate

    @property
    def channels(self) -> int:
        return self.model.channels

    @property
    def cardinality(self):
        """Cardinality of each codebook.
        """
        if self.extend_cardinality:
            return self.model.cardinality * self.num_virtual_steps
        else:
            return self.model.cardinality

    def forward(self, x: torch.Tensor) -> qt.QuantizedResult:
        raise NotImplementedError("Not supported, use encode and decode.")

    def encode(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]:
        indices, scales = self.model.encode(x)
        B, K, T = indices.shape
        indices = rearrange(indices, 'b (k v) t -> b k t v', k=self.codebooks_per_step)
        if self.extend_cardinality:
            for virtual_step in range(1, self.num_virtual_steps):
                indices[..., virtual_step] += self.model.cardinality * virtual_step
        indices = rearrange(indices, 'b k t v -> b k (t v)')
        return (indices, scales)

    def decode(self, codes: torch.Tensor, scale: tp.Optional[torch.Tensor] = None):
        B, K, T = codes.shape
        assert T % self.num_virtual_steps == 0
        codes = rearrange(codes, 'b k (t v) -> b (k v) t', v=self.num_virtual_steps)
        # We silently ignore potential errors from the LM when
        # using extend_cardinality.
        codes = codes % self.model.cardinality
        return self.model.decode(codes, scale)
