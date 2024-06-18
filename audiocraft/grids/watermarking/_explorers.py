# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import treetable as tt

from .._base_explorers import BaseExplorer


class WatermarkingMbExplorer(BaseExplorer):
    eval_metrics = ["acc", "bit_acc", "visqol", "fnr", "fpr", "sisnr"]

    def stages(self):
        return ["train", "valid", "valid_ema", "evaluate", "evaluate_ema"]

    def get_grid_meta(self):
        """Returns the list of Meta information to display for each XP/job."""
        return [
            tt.leaf("index", align=">"),
            tt.leaf("name", wrap=140),
            tt.leaf("state"),
            tt.leaf("sig", align=">"),
        ]

    def get_grid_metrics(self):
        """Return the metrics that should be displayed in the tracking table."""
        return [
            tt.group(
                "train",
                [
                    tt.leaf("epoch"),
                    tt.leaf("sisnr", ".3%"),
                    tt.leaf("wm_detection_identity", ".3%"),
                    tt.leaf("wm_mb_identity", ".3%"),
                ],
                align=">",
            ),
            tt.group(
                "valid",
                [
                    tt.leaf("sisnr", ".3%"),
                    tt.leaf("wm_detection_identity", ".3%"),
                    tt.leaf("wm_mb_identity", ".3%"),
                    # tt.leaf("loss_0", ".3%"),
                ],
                align=">",
            ),
            tt.group(
                "evaluate",
                [
                    tt.leaf("aug_identity_acc", ".4f"),
                    tt.leaf("aug_identity_fnr", ".4f"),
                    tt.leaf("aug_identity_fpr", ".4f"),
                    tt.leaf("aug_identity_bit_acc", ".4f"),
                    tt.leaf("pesq", ".4f"),
                    tt.leaf("all_aug_acc", ".4f"),
                    tt.leaf("localization_acc_padding", ".4f"),
                ],
                align=">",
            ),
        ]


class WatermarkingExplorer(BaseExplorer):
    eval_metrics = ["acc", "visqol", "fnr", "fpr", "sisnr"]

    def stages(self):
        return ["train", "valid", "valid_ema", "evaluate", "evaluate_ema"]

    def get_grid_meta(self):
        """Returns the list of Meta information to display for each XP/job."""
        return [
            tt.leaf("index", align=">"),
            tt.leaf("name", wrap=140),
            tt.leaf("state"),
            tt.leaf("sig", align=">"),
        ]

    def get_grid_metrics(self):
        """Return the metrics that should be displayed in the tracking table."""
        return [
            tt.group(
                "train",
                [
                    tt.leaf("epoch"),
                    tt.leaf("sisnr", ".3f"),
                    tt.leaf("wm_detection_identity"),
                ],
                align=">",
            ),
            tt.group(
                "valid",
                [
                    tt.leaf("sisnr", ".3f"),
                    tt.leaf("wm_detection_identity"),
                    # tt.leaf("loss_0", ".3%"),
                ],
                align=">",
            ),
            tt.group(
                "evaluate",
                [
                    tt.leaf("aug_identity_acc", ".4f"),
                    tt.leaf("aug_identity_fnr", ".4f"),
                    tt.leaf("aug_identity_fpr", ".4f"),
                    tt.leaf("pesq", ".4f"),
                    tt.leaf("all_aug_acc", ".4f"),
                    tt.leaf("localization_acc_padding", ".4f"),

                ],
                align=">",
            ),
        ]
