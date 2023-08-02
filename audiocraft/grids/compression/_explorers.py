# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import treetable as tt

from .._base_explorers import BaseExplorer


class CompressionExplorer(BaseExplorer):
    eval_metrics = ["sisnr", "visqol"]

    def stages(self):
        return ["train", "valid", "evaluate"]

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
                    tt.leaf("bandwidth", ".2f"),
                    tt.leaf("adv", ".4f"),
                    tt.leaf("d_loss", ".4f"),
                ],
                align=">",
            ),
            tt.group(
                "valid",
                [
                    tt.leaf("bandwidth", ".2f"),
                    tt.leaf("adv", ".4f"),
                    tt.leaf("msspec", ".4f"),
                    tt.leaf("sisnr", ".2f"),
                ],
                align=">",
            ),
            tt.group(
                "evaluate", [tt.leaf(name, ".3f") for name in self.eval_metrics], align=">"
            ),
        ]
