# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from audiocraft.quantization.vq import ResidualVectorQuantizer


class TestResidualVectorQuantizer:

    def test_rvq(self):
        x = torch.randn(1, 16, 2048)
        vq = ResidualVectorQuantizer(n_q=8, dimension=16, bins=8)
        res = vq(x, 1.)
        assert res.x.shape == torch.Size([1, 16, 2048])
