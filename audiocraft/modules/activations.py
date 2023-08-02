# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Callable


class CustomGLU(nn.Module):
    """Custom Gated Linear Unit activation.
    Applies a modified gated linear unit :math:`a * f(b)` where :math:`a` is the first half
    of the input matrices, :math:`b` is the second half, and :math:`f` is a provided activation
    function (i.e. sigmoid, swish, etc.).

    Args:
        activation (nn.Module): The custom activation to apply in the Gated Linear Unit
        dim (int): the dimension on which to split the input. Default: -1

    Shape:
        - Input: :math:`(\ast_1, N, \ast_2)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(\ast_1, M, \ast_2)` where :math:`M=N/2`

    Examples::
        >>> m = CustomGLU(nn.Sigmoid())
        >>> input = torch.randn(4, 2)
        >>> output = m(input)
    """
    def __init__(self, activation: nn.Module, dim: int = -1):
        super(CustomGLU, self).__init__()
        self.dim = dim
        self.activation = activation

    def forward(self, x: Tensor):
        assert x.shape[self.dim] % 2 == 0  # M = N / 2
        a, b = torch.chunk(x, 2, dim=self.dim)
        return a * self.activation(b)


class SwiGLU(CustomGLU):
    """SiLU Gated Linear Unit activation.
    Applies SiLU Gated Linear Unit :math:`a * SiLU(b)` where :math:`a` is
    the first half of the input matrices, :math:`b` is the second half.

    Args:
        dim (int): the dimension on which to split the input. Default: -1
    """
    def __init__(self, dim: int = -1):
        super(SwiGLU, self).__init__(nn.SiLU(), dim)


class GeGLU(CustomGLU):
    """GeLU Gated Linear Unit activation.
    Applies GeLU Gated Linear Unit :math:`a * GELU(b)` where :math:`a` is
    the first half of the input matrices, :math:`b` is the second half.

    Args:
        dim (int): the dimension on which to split the input. Default: -1
    """
    def __init__(self, dim: int = -1):
        super(GeGLU, self).__init__(nn.GELU(), dim)


class ReGLU(CustomGLU):
    """ReLU Gated Linear Unit activation.
    Applies ReLU Gated Linear Unit :math:`a * ReLU(b)` where :math:`a` is
    the first half of the input matrices, :math:`b` is the second half.

    Args:
        dim (int): the dimension on which to split the input. Default: -1
    """
    def __init__(self, dim: int = -1):
        super(ReGLU, self).__init__(nn.ReLU(), dim)


def get_activation_fn(
    activation: Union[str, Callable[[Tensor], Tensor]]
) -> Union[str, Callable[[Tensor], Tensor]]:
    """Helper function to map an activation string to the activation class.
    If the supplied activation is not a string that is recognized, the activation is passed back.

    Args:
        activation (str, or Callable[[Tensor], Tensor]): Activation to check
    """
    if isinstance(activation, str):
        if activation == "reglu":
            return ReGLU()
        elif activation == "geglu":
            return GeGLU()
        elif activation == "swiglu":
            return SwiGLU()
    return activation
