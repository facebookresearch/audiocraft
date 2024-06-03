# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import julius
import pesq
import torch
from audiocraft.metrics.pesq import PesqMetric
from ..common_utils import TempDirMixin, get_batch_white_noise


def tensor_pesq(y_pred: torch.Tensor, y: torch.Tensor, sr: int):
    # pesq returns error if no speech is detected, so we catch it
    if sr != 16000:
        y_pred = julius.resample_frac(y_pred, sr, 16000)
        y = julius.resample_frac(y, sr, 16000)
    P, n = 0, 0
    for ii in range(y_pred.size(0)):
        try:  # torchmetrics crashes when there is one error in the batch so doing it manually..
            P += pesq.pesq(16000, y[ii, 0].cpu().numpy(), y_pred[ii, 0].cpu().numpy())
            n += 1
        except pesq.NoUtterancesError:  # this error can append when the sample don't contain speech
            pass
    p = P / n if n != 0 else 0.0
    return p


class TestPesq(TempDirMixin):

    def test(self):
        sample_rate = 16_000
        duration = 20
        channel = 1
        bs = 10
        wavs = get_batch_white_noise(bs, channel, int(sample_rate * duration))

        pesq_metric = PesqMetric(sample_rate=sample_rate)
        pesq1 = pesq_metric(wavs, wavs)
        print(f"Pesq between 2 identical white noises: {pesq1}")
        assert pesq1 > 1

        pesq2 = tensor_pesq(wavs, wavs, 16000)
        assert torch.allclose(pesq1, torch.tensor(pesq2))
