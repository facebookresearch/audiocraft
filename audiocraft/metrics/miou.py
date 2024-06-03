# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def calculate_miou(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate the mean Intersection over Union (mIoU) between two binary tensors using PyTorch.

    Args:
        y_pred (torch.Tensor): Predicted binary tensor of shape [bsz, frames].
        y_true (torch.Tensor): Ground truth binary tensor of shape [bsz, frames].

    Returns:
        float: The mean Intersection over Union (mIoU) score.

    Reference:
        The Intersection over Union (IoU) metric is commonly used in computer vision.
        For more information, refer to the following paper:
        "SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation"
        by Vijay Badrinarayanan, Alex Kendall, Roberto Cipolla
    """
    # Ensure y_pred and y_true have the same shape
    if y_pred.shape != y_true.shape:
        raise ValueError("Input tensors must have the same shape")

    # converting predictions to binary vector
    y_pred = y_pred > 0.5
    # Compute the intersection and union
    intersection = torch.logical_and(y_pred, y_true)
    union = torch.logical_or(y_pred, y_true)

    # Compute IoU for each sample in the batch
    iou_per_sample = torch.sum(intersection, dim=1) / torch.sum(union, dim=1)
    # Calculate mIoU by taking the mean across the batch
    miou = torch.mean(iou_per_sample).item()

    return miou
