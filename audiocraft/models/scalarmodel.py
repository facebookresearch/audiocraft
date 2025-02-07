"""This lobe enables the integration of speech codec model (SQ-Codec) with scalar quantization,.

SQ-Codec effectively maps the complex speech signal into a finite and compact latent space, named scalar latent space.

Repository: https://github.com/yangdongchao/SimpleSpeech
Paper: https://arxiv.org/abs/2406.02328, https://arxiv.org/abs/2408.13893

"""

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf
from torch.autograd import Function
from torch.nn.utils import remove_weight_norm, weight_norm


class ScalarModel(nn.Module):
    """
    A custom neural network model for encoding and decoding audio signals.

    The model consists of an encoder-decoder architecture with optional
    causal convolutions, downsampling, and upsampling layers. It uses
    vector quantization and various convolutional blocks for processing.


    Arguments
    ---------
    num_bands : int
        Number of input bands (or channels).
    sample_rate : int
        Sample rate of the input signal.
    causal : bool
        If True, uses causal convolutions for processing.
    num_samples : int
        Number of samples to process for downsampling or upsampling.
    downsample_factors : list of int
        List of factors to downsample the input.
    downsample_kernel_sizes : list of int
        List of kernel sizes for downsampling layers.
    upsample_factors : list of int
        List of factors to upsample the input.
    upsample_kernel_sizes : list of int
        List of kernel sizes for upsampling layers.
    latent_hidden_dim : int
        Dimension of the latent representation.
    default_kernel_size : int
        Default kernel size for convolutional layers.
    delay_kernel_size : int
        Kernel size used for the delay convolutional layer.
    init_channel : int
        Number of initial channels for the encoder and decoder.
    res_kernel_size : int
        Kernel size used for the residual convolutional blocks.

    Example
    -------
    >>> model = ScalarModel(num_bands=1, sample_rate=16000,causal=True,num_samples=2,downsample_factors=[2,4,4,5],downsample_kernel_sizes=[4,8,8,10],upsample_factors=[5,4,4,2],upsample_kernel_sizes=[10,8,8,4],latent_hidden_dim=36,default_kernel_size=7,delay_kernel_size=5,init_channel=48,res_kernel_size=7) # doctest: +SKIP
    >>> audio = torch.randn(3, 1, 16000)
    >>> quant_emb = model.encode(audio) # doctest: +SKIP
    >>> quant_emb.shape
    torch.Size([3, 36, 50])
    >>> rec = model.decode(quant_emb) # doctest: +SKIP
    >>> rec.shap) # doctest: +SKIP
    torch.Size([3, 1, 16000])
    """

    def __init__(
        self,
        num_bands=1,
        sample_rate=16000,
        causal=True,
        num_samples=2,
        downsample_factors=[2,4,4,5],
        downsample_kernel_sizes=[4,8,8,10],
        upsample_factors=[5,4,4,2],
        upsample_kernel_sizes=[10,8,8,4],
        latent_hidden_dim=36,
        default_kernel_size=7,
        delay_kernel_size=5,
        init_channel=48,
        res_kernel_size=7,
    ):
        super(ScalarModel, self).__init__()
        self.sample_rate = sample_rate
        self.encoder = []
        self.decoder = []
        self.vq = lambda x: CustomRoundingFunction.apply(x, "binary")

        # Encoder layers
        self.encoder.append(
            weight_norm(
                Conv1d(
                    num_bands,
                    init_channel,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        )
        if num_samples > 1:
            # Downsampling layer
            self.encoder.append(
                PreProcessor(
                    init_channel,
                    init_channel,
                    num_samples,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        for i, down_factor in enumerate(downsample_factors):
            self.encoder.append(
                ResEncoderBlock(
                    init_channel * np.power(2, i),
                    init_channel * np.power(2, i + 1),
                    down_factor,
                    downsample_kernel_sizes[i],
                    res_kernel_size,
                    causal=causal,
                )
            )
        self.encoder.append(
            weight_norm(
                Conv1d(
                    init_channel * np.power(2, len(downsample_factors)),
                    latent_hidden_dim,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        )

        # Decoder layers
        self.decoder.append(
            weight_norm(
                Conv1d(
                    latent_hidden_dim,
                    init_channel * np.power(2, len(upsample_factors)),
                    kernel_size=delay_kernel_size,
                )
            )
        )
        for i, upsample_factor in enumerate(upsample_factors):
            self.decoder.append(
                ResDecoderBlock(
                    init_channel * np.power(2, len(upsample_factors) - i),
                    init_channel * np.power(2, len(upsample_factors) - i - 1),
                    upsample_factor,
                    upsample_kernel_sizes[i],
                    res_kernel_size,
                    causal=causal,
                )
            )
        if num_samples > 1:
            self.decoder.append(
                PostProcessor(
                    init_channel,
                    init_channel,
                    num_samples,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        self.decoder.append(
            weight_norm(
                Conv1d(
                    init_channel,
                    num_bands,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        )

        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)

    def forward(self, x):
        """
        Performs a forward pass through the encoder and decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, length).

        Returns
        -------
        torch.Tensor
            Reconstructed output tensor.
        """
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x))
        x = self.vq(x)  # Quantization step
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return x

    def inference(self, x):
        """
        Encodes input tensor `x` and decodes the quantized embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, length).

        Returns
        -------
        tuple
            A tuple (emb, emb_quant, x), where `emb` is the latent embedding,
            `emb_quant` is the quantized embedding, and `x` is the decoded output.
        """
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x))
        emb = x
        emb_quant = self.vq(emb)
        x = emb_quant
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return emb, emb_quant, x

    def encode(self, x):
        """
        Encodes the input tensor `x` into a quantized embedding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, length).

        Returns
        -------
        torch.Tensor
            Quantized embedding.
        """
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = F.tanh(layer(x))
        emb = x
        emb_quant = self.vq(emb)
        return emb_quant

    def decode(self, emb_quant):
        """
        Decodes the quantized embeddings back into a tensor.

        Parameters
        ----------
        emb_quant : torch.Tensor
            Quantized embedding tensor.

        Returns
        -------
        torch.Tensor
            Reconstructed output tensor.
        """
        x = emb_quant
        for i, layer in enumerate(self.decoder):
            x = layer(x)
        return x


class CustomRoundingFunction(Function):
    """
    A customizable rounding function for various rounding operations, including:
    - Rounding to the nearest multiple of a specified divisor.
    - Rounding to the nearest integer.
    - Applying the Heaviside step function.

    Arguments
    ---------
    mode : str
        The mode of the operation. Can be 'round', 'binary', or 'heaviside'.
    divisor : float, optional
        The divisor for rounding. Only used in 'round' mode.
    """

    @staticmethod
    def forward(ctx, input, mode="round", divisor=1.0):
        """
        Forward pass for the custom rounding function.

        Arguments
        ---------
        ctx : context object
            Context object used to store information for the backward computation.
        input : torch.Tensor
            The input tensor to be processed.
        mode : str
            The mode of the operation ('round', 'binary', 'heaviside').
        divisor : float
            The divisor for rounding. Only used in 'round' mode.

        Returns
        -------
        torch.Tensor
            The processed tensor after applying the operation.
        """
        ctx.mode = mode
        ctx.divisor = divisor

        if mode == "round":
            return torch.round(divisor * input) / divisor
        elif mode == "binary":
            return torch.round(input)
        elif mode == "heaviside":
            values = torch.tensor([0.0]).type_as(input)
            return torch.heaviside(input, values)
        else:
            raise ValueError(
                f"Invalid mode '{mode}'. Supported modes: 'round', 'binary', 'heaviside'."
            )

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the custom rounding function.

        Arguments
        ---------
        ctx : context object
            Context object containing information saved during the forward pass.
        grad_output : torch.Tensor
            The gradient of the output with respect to the loss.

        Returns
        -------
        torch.Tensor
            The gradient of the input with respect to the loss.
        """
        # For all modes, the gradient is propagated unchanged.
        return grad_output.clone(), None, None


class PreProcessor(nn.Module):
    """
    A module for preprocessing input data through convolution and pooling operations.
    It is used as an initial step before the encoder blocks in the ScalarModel, particularly when the kernel_size for average pooling operation exceeds 1.

    Arguments
    ---------
    n_in : int
        Number of input channels.
    n_out : int
        Number of output channels.
    num_samples : int
        Number of samples for pooling.
    kernel_size : int, optional
        Size of the convolutional kernel (default is 7).
    causal : bool, optional
        If True, applies causal convolution (default is False).
    """

    def __init__(self, n_in, n_out, num_samples, kernel_size=7, causal=False):
        super(PreProcessor, self).__init__()
        self.pooling = torch.nn.AvgPool1d(kernel_size=num_samples)
        self.conv = Conv1d(n_in, n_out, kernel_size=kernel_size, causal=causal)
        self.activation = nn.PReLU()

    def forward(self, x):
        """
        Applies convolution, activation, and pooling to the input data.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Processed output tensor.
        """
        output = self.activation(self.conv(x))
        output = self.pooling(output)
        return output


class PostProcessor(nn.Module):
    """
    A module for postprocessing data through convolution and reshaping.
    It is used as an initial step after the decoder blocks in the ScalarModel, particularly when the kernel_size for average pooling operation exceeds 1.

    Arguments
    ---------
    n_in : int
        Number of input channels.
    n_out : int
        Number of output channels.
    num_samples : int
        Number of samples for repetition.
    kernel_size : int, optional
        Size of the convolutional kernel (default is 7).
    causal : bool, optional
        If True, applies causal convolution (default is False).
    """

    def __init__(self, n_in, n_out, num_samples, kernel_size=7, causal=False):
        super(PostProcessor, self).__init__()
        self.num_samples = num_samples
        self.conv = Conv1d(n_in, n_out, kernel_size=kernel_size, causal=causal)
        self.activation = nn.PReLU()

    def forward(self, x):
        """
        Applies reshaping, repetition, and convolution to the input data.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Processed output tensor.
        """
        x = torch.transpose(x, 1, 2)
        B, T, C = x.size()
        x = x.repeat(1, 1, self.num_samples).view(B, -1, C)
        x = torch.transpose(x, 1, 2)
        output = self.activation(self.conv(x))
        return output


class DownsampleLayer(nn.Module):
    """
    A downsampling layer that applies convolution, optional pooling, and activation.

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel.
    stride : int, optional
        Stride of the convolution (default is 1).
    causal : bool, optional
        If True, applies causal convolution (default is False).
    activation : nn.Module, optional
        Activation function (default is PReLU).
    use_weight_norm : bool, optional
        If True, applies weight normalization to the convolution (default is True).
    pooling : bool, optional
        If True, applies an average pooling operation (default is False).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        activation=nn.PReLU(),
        use_weight_norm: bool = True,
        pooling: bool = False,
    ):
        super(DownsampleLayer, self).__init__()
        self.pooling = pooling
        self.stride = stride
        self.activation = activation
        self.use_weight_norm = use_weight_norm
        if pooling:
            self.layer = Conv1d(
                in_channels, out_channels, kernel_size, causal=causal
            )
            self.pooling = nn.AvgPool1d(kernel_size=stride)
        else:
            self.layer = Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                causal=causal,
            )
        if use_weight_norm:
            self.layer = weight_norm(self.layer)

    def forward(self, x):
        """
        Applies convolution, optional pooling, and activation to the input data.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Processed output tensor.
        """
        x = self.layer(x)
        x = self.activation(x) if self.activation is not None else x
        if self.pooling:
            x = self.pooling(x)
        return x

    def remove_weight_norm(self):
        """
        Removes weight normalization from the convolutional layer.
        """
        if self.use_weight_norm:
            remove_weight_norm(self.layer)


class UpsampleLayer(nn.Module):
    """
    An upsampling layer that applies transposed convolution or repetition, with activation.

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel.
    stride : int, optional
        Stride of the transposed convolution (default is 1).
    causal : bool, optional
        If True, applies causal convolution (default is False).
    activation : nn.Module, optional
        Activation function (default is PReLU).
    use_weight_norm : bool, optional
        If True, applies weight normalization to the convolution (default is True).
    repeat : bool, optional
        If True, applies repetition instead of transposed convolution (default is False).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        activation=nn.PReLU(),
        use_weight_norm: bool = True,
        repeat: bool = False,
    ):
        super(UpsampleLayer, self).__init__()
        self.repeat = repeat
        self.stride = stride
        self.activation = activation
        self.use_weight_norm = use_weight_norm
        if repeat:
            self.layer = Conv1d(
                in_channels, out_channels, kernel_size, causal=causal
            )
        else:
            self.layer = ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                causal=causal,
            )
        if use_weight_norm:
            self.layer = weight_norm(self.layer)

    def forward(self, x):
        """
        Applies upsampling through transposed convolution or repetition, followed by activation.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Processed output tensor.
        """
        x = self.layer(x)
        x = self.activation(x) if self.activation is not None else x
        if self.repeat:
            x = torch.transpose(x, 1, 2)
            B, T, C = x.size()
            x = x.repeat(1, 1, self.stride).view(B, -1, C)
            x = torch.transpose(x, 1, 2)
        return x

    def remove_weight_norm(self):
        """
        Removes weight normalization from the convolutional layer.
        """
        if self.use_weight_norm:
            remove_weight_norm(self.layer)


class ResidualUnit(nn.Module):
    """
    A residual unit with two convolutional layers and activation functions.
    This module is commonly used in the encoder and decoder blocks of the ScalarModel

    Arguments
    ---------
    n_in : int
        Number of input channels.
    n_out : int
        Number of output channels.
    dilation : int
        Dilation factor for the first convolutional layer.
    res_kernel_size : int, optional
        Size of the convolutional kernel for residual connections (default is 7).
    causal : bool, optional
        If True, applies causal convolution (default is False).
    """

    def __init__(self, n_in, n_out, dilation, res_kernel_size=7, causal=False):
        super(ResidualUnit, self).__init__()
        self.conv1 = weight_norm(
            Conv1d(
                n_in,
                n_out,
                kernel_size=res_kernel_size,
                dilation=dilation,
                causal=causal,
            )
        )
        self.conv2 = weight_norm(
            Conv1d(n_in, n_out, kernel_size=1, causal=causal)
        )
        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()

    def forward(self, x):
        """
        Applies two convolutional layers with activations and adds the input for a residual connection.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor with residual connection applied.
        """
        output = self.activation1(self.conv1(x))
        output = self.activation2(self.conv2(output))
        return output + x


class ResEncoderBlock(nn.Module):
    """
    A residual encoder block with multiple residual units and a downsampling layer.

    Arguments
    ---------
    n_in : int
        Number of input channels.
    n_out : int
        Number of output channels.
    stride : int
        Stride for the downsampling layer.
    down_kernel_size : int
        Kernel size for the downsampling layer.
    res_kernel_size : int, optional
        Size of the convolutional kernel for residual connections (default is 7).
    causal : bool, optional
        If True, applies causal convolution (default is False).
    """

    def __init__(
        self,
        n_in,
        n_out,
        stride,
        down_kernel_size,
        res_kernel_size=7,
        causal=False,
    ):
        super(ResEncoderBlock, self).__init__()
        self.convs = nn.ModuleList(
            [
                ResidualUnit(
                    n_in,
                    n_out // 2,
                    dilation=1,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out // 2,
                    n_out // 2,
                    dilation=3,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out // 2,
                    n_out // 2,
                    dilation=5,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out // 2,
                    n_out // 2,
                    dilation=7,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out // 2,
                    n_out // 2,
                    dilation=9,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
            ]
        )
        self.down_conv = DownsampleLayer(
            n_in, n_out, down_kernel_size, stride=stride, causal=causal
        )

    def forward(self, x):
        """
        Applies a series of residual units and a downsampling layer.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Processed output tensor.
        """
        for conv in self.convs:
            x = conv(x)
        x = self.down_conv(x)
        return x


class ResDecoderBlock(nn.Module):
    """
    A residual decoder block with upsampling and multiple residual units.

    Arguments
    ---------
    n_in : int
        Number of input channels.
    n_out : int
        Number of output channels.
    stride : int
        Stride for the upsampling layer.
    up_kernel_size : int
        Kernel size for the upsampling layer.
    res_kernel_size : int, optional
        Size of the convolutional kernel for residual connections (default is 7).
    causal : bool, optional
        If True, applies causal convolution (default is False).
    """

    def __init__(
        self,
        n_in,
        n_out,
        stride,
        up_kernel_size,
        res_kernel_size=7,
        causal=False,
    ):
        super(ResDecoderBlock, self).__init__()
        self.up_conv = UpsampleLayer(
            n_in,
            n_out,
            kernel_size=up_kernel_size,
            stride=stride,
            causal=causal,
            activation=None,
        )
        self.convs = nn.ModuleList(
            [
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=1,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=3,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=5,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=7,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
                ResidualUnit(
                    n_out,
                    n_out,
                    dilation=9,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                ),
            ]
        )

    def forward(self, x):
        """
        Applies upsampling followed by a series of residual units.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Processed output tensor.
        """
        x = self.up_conv(x)
        for conv in self.convs:
            x = conv(x)
        return x


class Conv1d(nn.Conv1d):
    """
    Custom 1D convolution layer with an optional causal mode.

    This class extends PyTorch's `nn.Conv1d` and allows for causal convolutions
    by automatically applying the correct amount of padding to ensure that the output
    does not depend on future inputs, which is useful for sequential data processing.

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel.
    stride : int, optional
        Stride of the convolution (default is 1).
    dilation : int, optional
        Dilation factor for the convolution (default is 1).
    groups : int, optional
        Number of blocked connections from input channels to output channels (default is 1).
    padding_mode : str, optional
        Padding mode to use ('zeros', 'reflect', 'replicate', or 'circular') (default is 'zeros').
    bias : bool, optional
        If True, adds a learnable bias to the output (default is True).
    padding : int, optional
        Explicit padding value. If not provided, it will be computed automatically.
    causal : bool, optional
        If True, applies causal convolution where the output depends only on the past and current inputs (default is False).
    w_init_gain : str, optional
        Gain value used for Xavier initialization (e.g., 'relu', 'tanh', etc.). If provided, applies Xavier uniform initialization to the convolutional weights.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
        padding=None,
        causal: bool = False,
        w_init_gain=None,
    ):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = get_padding(kernel_size, dilation)
        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias,
        )
        if w_init_gain is not None:
            torch.nn.init.xavier_uniform_(
                self.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
            )

    def forward(self, x):
        """
        Applies the forward pass of the convolutional layer.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, sequence_length).

        Returns
        -------
        torch.Tensor
            The output tensor after applying the convolution operation.
            If `causal` is True, the input tensor is padded to ensure that
            the output at each timestep only depends on the current and previous inputs.
        """
        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        return super(Conv1d, self).forward(x)


class ConvTranspose1d(nn.ConvTranspose1d):
    """
    Custom transposed 1D convolution layer with causal option.

    Arguments
    ---------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Size of the convolutional kernel.
    stride : int, optional
        Stride of the convolution (default is 1).
    output_padding : int, optional
        Additional size added to one side of the output (default is 0).
    groups : int, optional
        Number of blocked connections (default is 1).
    bias : bool, optional
        If True, adds a learnable bias (default is True).
    dilation : int, optional
        Dilation factor (default is 1).
    padding : int, optional
        Explicit padding value (default is None).
    padding_mode : str, optional
        Padding mode (default is 'zeros').
    causal : bool, optional
        If True, applies causal convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding=None,
        padding_mode: str = "zeros",
        causal: bool = False,
    ):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert (
                padding == 0
            ), "padding is not allowed in causal ConvTranspose1d."
            assert (
                kernel_size == 2 * stride
            ), "kernel_size must be equal to 2*stride is not allowed in causal ConvTranspose1d."
        super(ConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.causal = causal
        self.stride = stride

    def forward(self, x):
        """
        Applies the transposed convolution operation.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Transposed convolved output tensor.
        """
        x = super(ConvTranspose1d, self).forward(x)
        if self.causal:
            x = x[:, :, : -self.stride]
        return x


def decimal_to_ternary_matrix(decimals, D):
    """
    Convert a tensor of decimal numbers to a D*T ternary matrix for each batch.

    Arguments
    ---------
    decimals : torch.Tensor
        A 2D tensor of decimal numbers with shape (B, T), where B is the batch size
        and T is the number of elements in each batch.
    D : int
        Number of ternary digits to represent each number (depth).

    Returns
    -------
    torch.Tensor
        A 3D tensor of shape (B, D, T) where each slice along the first dimension
        corresponds to a batch, and each column is represented as a ternary number.
    """
    B, T = decimals.shape
    ternary_matrix = torch.zeros((B, D, T), dtype=torch.long)
    for pos in range(D):
        ternary_matrix[:, pos, :] = decimals % 3  # Modulo operation
        decimals //= 3  # Floor division for next ternary digit

    return ternary_matrix


def ternary_matrix_to_decimal(matrix):
    """
    Convert a B*D*N ternary matrix to a 2D array of decimal numbers for each batch.

    Arguments
    ---------
    matrix : numpy.ndarray
        A 3D numpy array of shape (B, D, N), where B is the batch size, D is the number
        of ternary digits, and N is the number of ternary numbers in each batch.

    Returns
    -------
    numpy.ndarray
        A 2D numpy array of shape (B, N), where each value represents the decimal
        equivalent of the corresponding ternary number in the input matrix.
    """
    (
        B,
        D,
        N,
    ) = (
        matrix.shape
    )  # B is the batch size, D is the number of digits, N is the number of ternary numbers
    powers_of_three = 3 ** np.arange(D)  # [3^0, 3^1, ..., 3^(D-1)]

    # Reshape powers_of_three for broadcasting: [D] -> [1, D, 1]
    powers_of_three = powers_of_three[:, np.newaxis]  # Shape [D, 1]

    # Compute dot product using broadcasting: matrix * powers_of_three along D axis
    decimals = np.sum(matrix * powers_of_three, axis=1)  # Sum along the D axis

    return decimals


def get_padding(kernel_size, dilation=1):
    """
    Computes the padding size for a given kernel size and dilation.

    Arguments
    ---------
    kernel_size : int
        Size of the convolutional kernel.
    dilation : int, optional
        Dilation factor for convolution (default is 1).

    Returns
    -------
    int
        Calculated padding size.
    """
    return int((kernel_size * dilation - dilation) / 2)