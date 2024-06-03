# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal

import torch
import torch.nn as nn


class WMDetectionLoss(nn.Module):
    """Compute the detection loss"""
    def __init__(self, p_weight: float = 1.0, n_weight: float = 1.0) -> None:
        super().__init__()
        self.criterion = nn.NLLLoss()
        self.p_weight = p_weight
        self.n_weight = n_weight

    def forward(self, positive, negative, mask, message=None):

        positive = positive[:, :2, :]  # b 2+nbits t -> b 2 t
        negative = negative[:, :2, :]  # b 2+nbits t -> b 2 t

        # dimensionality of positive [bsz, classes=2, time_steps]
        # correct classes for pos = [bsz, time_steps] where all values = 1 for positive
        classes_shape = positive[
            :, 0, :
        ]  # same as positive or negative but dropping dim=1
        pos_correct_classes = torch.ones_like(classes_shape, dtype=int)
        neg_correct_classes = torch.zeros_like(classes_shape, dtype=int)

        # taking log because network outputs softmax
        # NLLLoss expects a logsoftmax input
        positive = torch.log(positive)
        negative = torch.log(negative)

        if not torch.all(mask == 1):
            # pos_correct_classes [bsz, timesteps] mask [bsz, 1, timesptes]
            # mask is applied to the watermark, this basically flips the tgt class from 1 (positive)
            # to 0 (negative) in the correct places
            pos_correct_classes = pos_correct_classes * mask[:, 0, :].to(int)
            loss_p = self.p_weight * self.criterion(positive, pos_correct_classes)
            # no need for negative class loss here since some of the watermark
            # is masked to negative
            return loss_p

        else:
            loss_p = self.p_weight * self.criterion(positive, pos_correct_classes)
            loss_n = self.n_weight * self.criterion(negative, neg_correct_classes)
            return loss_p + loss_n


class WMMbLoss(nn.Module):
    def __init__(self, temperature: float, loss_type: Literal["bce", "mse"]) -> None:
        """
        Compute the masked sample-level detection loss
        (https://arxiv.org/pdf/2401.17264)

        Args:
            temperature: temperature for loss computation
            loss_type: bce or mse between outputs and original message
        """
        super().__init__()
        self.bce_with_logits = (
            nn.BCEWithLogitsLoss()
        )  # same as Softmax + NLLLoss, but when only 1 output unit
        self.mse = nn.MSELoss()
        self.loss_type = loss_type
        self.temperature = temperature

    def forward(self, positive, negative, mask, message):
        """
        Compute decoding loss
        Args:
            positive: outputs on watermarked samples [bsz, 2+nbits, time_steps]
            negative: outputs on not watermarked samples [bsz, 2+nbits, time_steps]
            mask: watermark mask [bsz, 1, time_steps]
            message: original message [bsz, nbits] or None
        """
        # # no use of negative at the moment
        # negative = negative[:, 2:, :]  # b 2+nbits t -> b nbits t
        # negative = torch.masked_select(negative, mask)
        if message.size(0) == 0:
            return torch.tensor(0.0)
        positive = positive[:, 2:, :]  # b 2+nbits t -> b nbits t
        assert (
            positive.shape[-2] == message.shape[1]
        ), "in decoding loss: \
            enc and dec don't share nbits, are you using multi-bit?"

        # cut last dim of positive to keep only where mask is 1
        new_shape = [*positive.shape[:-1], -1]  # b nbits -1
        positive = torch.masked_select(positive, mask == 1).reshape(new_shape)

        message = message.unsqueeze(-1).repeat(1, 1, positive.shape[2])  # b k -> b k t
        if self.loss_type == "bce":
            # in this case similar to temperature in softmax
            loss = self.bce_with_logits(positive / self.temperature, message.float())
        elif self.loss_type == "mse":
            loss = self.mse(positive / self.temperature, message.float())

        return loss
