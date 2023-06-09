# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch

from audiocraft.modules.lstm import StreamableLSTM


class TestStreamableLSTM:

    def test_lstm(self):
        B, C, T = 4, 2, random.randint(1, 100)

        lstm = StreamableLSTM(C, 3, skip=False)
        x = torch.randn(B, C, T)
        y = lstm(x)

        print(y.shape)
        assert y.shape == torch.Size([B, C, T])

    def test_lstm_skip(self):
        B, C, T = 4, 2, random.randint(1, 100)

        lstm = StreamableLSTM(C, 3, skip=True)
        x = torch.randn(B, C, T)
        y = lstm(x)

        assert y.shape == torch.Size([B, C, T])
