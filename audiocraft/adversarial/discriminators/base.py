# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
import typing as tp

import torch
import torch.nn as nn


FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
MultiDiscriminatorOutputType = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]


class MultiDiscriminator(ABC, nn.Module):
    """Base implementation for discriminators composed of sub-discriminators acting at different scales.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> MultiDiscriminatorOutputType:
        ...

    @property
    @abstractmethod
    def num_discriminators(self) -> int:
        """Number of discriminators.
        """
        ...
