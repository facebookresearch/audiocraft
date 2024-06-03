# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import julius
import pesq

import torch
import torchmetrics


class PesqMetric(torchmetrics.Metric):
    """Metric for Perceptual Evaluation of Speech Quality.
    (https://doi.org/10.5281/zenodo.6549559)

    """

    sum_pesq: torch.Tensor
    total: torch.Tensor

    def __init__(self, sample_rate: int):
        super().__init__()
        self.sr = sample_rate

        self.add_state("sum_pesq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if self.sr != 16000:
            preds = julius.resample_frac(preds, self.sr, 16000)
            targets = julius.resample_frac(targets, self.sr, 16000)
        for ii in range(preds.size(0)):
            try:
                self.sum_pesq += pesq.pesq(
                    16000, targets[ii, 0].detach().cpu().numpy(), preds[ii, 0].detach().cpu().numpy()
                )
                self.total += 1
            except (
                pesq.NoUtterancesError
            ):  # this error can append when the sample don't contain speech
                pass

    def compute(self) -> torch.Tensor:
        return (
            self.sum_pesq / self.total
            if (self.total != 0).item()
            else torch.tensor(0.0)
        )
