# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import treetable as tt

from .._base_explorers import BaseExplorer


class LMExplorer(BaseExplorer):
    eval_metrics: tp.List[str] = []

    def stages(self) -> tp.List[str]:
        return ['train', 'valid']

    def get_grid_metrics(self):
        """Return the metrics that should be displayed in the tracking table."""
        return [
            tt.group(
                'train',
                [
                    tt.leaf('epoch'),
                    tt.leaf('duration', '.1f'),  # duration in minutes
                    tt.leaf('ping'),
                    tt.leaf('ce', '.4f'),  # cross entropy
                    tt.leaf("ppl", '.3f'),  # perplexity
                ],
                align='>',
            ),
            tt.group(
                'valid',
                [
                    tt.leaf('ce', '.4f'),
                    tt.leaf('ppl', '.3f'),
                    tt.leaf('best_ppl', '.3f'),
                ],
                align='>',
            ),
        ]

    def process_sheep(self, sheep, history):
        parts = super().process_sheep(sheep, history)

        track_by = {'ppl': 'lower'}  # values should be in ['lower', 'higher']
        best_metrics = {k: (1 if v == 'lower' else -1) * float('inf') for k, v in track_by.items()}

        def comparator(mode, a, b):
            return a < b if mode == 'lower' else a > b

        for metrics in history:
            for key, sub in metrics.items():
                for metric in track_by:
                    # for the validation set, keep track of best metrics (ppl in this example)
                    # this is so we can conveniently compare metrics between runs in the grid
                    if key == 'valid' and metric in sub and comparator(
                        track_by[metric], sub[metric], best_metrics[metric]
                    ):
                        best_metrics[metric] = sub[metric]

        if 'valid' in parts:
            parts['valid'].update({f'best_{k}': v for k, v in best_metrics.items()})
        return parts


class GenerationEvalExplorer(BaseExplorer):
    eval_metrics: tp.List[str] = []

    def stages(self) -> tp.List[str]:
        return ['evaluate']

    def get_grid_metrics(self):
        """Return the metrics that should be displayed in the tracking table."""
        return [
            tt.group(
                'evaluate',
                [
                    tt.leaf('epoch', '.3f'),
                    tt.leaf('duration', '.1f'),
                    tt.leaf('ping'),
                    tt.leaf('ce', '.4f'),
                    tt.leaf('ppl', '.3f'),
                    tt.leaf('fad', '.3f'),
                    tt.leaf('kld', '.3f'),
                    tt.leaf('text_consistency', '.3f'),
                    tt.leaf('chroma_cosine', '.3f'),
                ],
                align='>',
            ),
        ]
