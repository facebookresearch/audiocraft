# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchmetrics

from ..data.audio_utils import convert_audio
from ..modules.chroma import ChromaExtractor


class ChromaCosineSimilarityMetric(torchmetrics.Metric):
    """Chroma cosine similarity metric.

    This metric extracts a chromagram for a reference waveform and
    a generated waveform and compares each frame using the cosine similarity
    function. The output is the mean cosine similarity.

    Args:
        sample_rate (int): Sample rate used by the chroma extractor.
        n_chroma (int): Number of chroma used by the chroma extractor.
        radix2_exp (int): Exponent for the chroma extractor.
        argmax (bool): Whether the chroma extractor uses argmax.
        eps (float): Epsilon for cosine similarity computation.
    """
    def __init__(self, sample_rate: int, n_chroma: int, radix2_exp: int, argmax: bool, eps: float = 1e-8):
        super().__init__()
        self.chroma_sample_rate = sample_rate
        self.n_chroma = n_chroma
        self.eps = eps
        self.chroma_extractor = ChromaExtractor(sample_rate=self.chroma_sample_rate, n_chroma=self.n_chroma,
                                                radix2_exp=radix2_exp, argmax=argmax)
        self.add_state("cosine_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("weight", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor,
               sizes: torch.Tensor, sample_rates: torch.Tensor) -> None:
        """Compute cosine similarity between chromagrams and accumulate scores over the dataset."""
        if preds.size(0) == 0:
            return

        assert preds.shape == targets.shape, (
            f"Preds and target shapes mismatch: preds={preds.shape}, targets={targets.shape}")
        assert preds.size(0) == sizes.size(0), (
            f"Number of items in preds ({preds.shape}) mismatch ",
            f"with sizes ({sizes.shape})")
        assert preds.size(0) == sample_rates.size(0), (
            f"Number of items in preds ({preds.shape}) mismatch ",
            f"with sample_rates ({sample_rates.shape})")
        assert torch.all(sample_rates == sample_rates[0].item()), "All sample rates are not the same in the batch"

        device = self.weight.device
        preds, targets = preds.to(device), targets.to(device)  # type: ignore
        sample_rate = sample_rates[0].item()
        preds = convert_audio(preds, from_rate=sample_rate, to_rate=self.chroma_sample_rate, to_channels=1)
        targets = convert_audio(targets, from_rate=sample_rate, to_rate=self.chroma_sample_rate, to_channels=1)
        gt_chroma = self.chroma_extractor(targets)
        gen_chroma = self.chroma_extractor(preds)
        chroma_lens = (sizes / self.chroma_extractor.winhop).ceil().int()
        for i in range(len(gt_chroma)):
            t = int(chroma_lens[i].item())
            cosine_sim = torch.nn.functional.cosine_similarity(
                gt_chroma[i, :t], gen_chroma[i, :t], dim=1, eps=self.eps)
            self.cosine_sum += cosine_sim.sum(dim=0)  # type: ignore
            self.weight += torch.tensor(t)  # type: ignore

    def compute(self) -> float:
        """Computes the average cosine similarty across all generated/target chromagrams pairs."""
        assert self.weight.item() > 0, "Unable to compute with total number of comparisons <= 0"  # type: ignore
        return (self.cosine_sum / self.weight).item()  # type: ignore
