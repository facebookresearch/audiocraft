# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from audiocraft.modules.activations import CustomGLU


class TestActivations:
    def test_custom_glu_calculation(self):

        activation = CustomGLU(nn.Identity())

        initial_shape = (4, 8, 8)

        part_a = torch.ones(initial_shape) * 2
        part_b = torch.ones(initial_shape) * -1
        input = torch.cat((part_a, part_b), dim=-1)

        output = activation(input)

        # ensure all dimensions match initial shape
        assert output.shape == initial_shape
        # ensure the gating was calculated correctly a * f(b)
        assert torch.all(output == -2).item()
