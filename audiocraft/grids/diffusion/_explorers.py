# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import treetable as tt

from .._base_explorers import BaseExplorer


class DiffusionExplorer(BaseExplorer):
    eval_metrics = ["sisnr", "visqol"]

    def stages(self):
        return ["train", "valid", "valid_ema", "evaluate", "evaluate_ema"]

    def get_grid_meta(self):
        """Returns the list of Meta information to display for each XP/job.
        """
        return [
            tt.leaf("index", align=">"),
            tt.leaf("name", wrap=140),
            tt.leaf("state"),
            tt.leaf("sig", align=">"),
        ]

    def get_grid_metrics(self):
        """Return the metrics that should be displayed in the tracking table.
        """
        return [
            tt.group(
                "train",
                [
                    tt.leaf("epoch"),
                    tt.leaf("loss", ".3%"),
                ],
                align=">",
            ),
            tt.group(
                "valid",
                [
                    tt.leaf("loss", ".3%"),
                    # tt.leaf("loss_0", ".3%"),
                ],
                align=">",
            ),
            tt.group(
                "valid_ema",
                [
                    tt.leaf("loss", ".3%"),
                    # tt.leaf("loss_0", ".3%"),
                ],
                align=">",
            ),
            tt.group(
                "evaluate", [tt.leaf("rvm", ".4f"), tt.leaf("rvm_0", ".4f"),
                             tt.leaf("rvm_1", ".4f"), tt.leaf("rvm_2", ".4f"),
                             tt.leaf("rvm_3", ".4f"), ], align=">"
            ),
            tt.group(
                "evaluate_ema", [tt.leaf("rvm", ".4f"), tt.leaf("rvm_0", ".4f"),
                                 tt.leaf("rvm_1", ".4f"), tt.leaf("rvm_2", ".4f"),
                                 tt.leaf("rvm_3", ".4f")], align=">"
            ),
        ]
