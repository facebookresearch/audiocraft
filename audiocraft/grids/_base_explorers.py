# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
import time
import typing as tp
from dora import Explorer
import treetable as tt


def get_sheep_ping(sheep) -> tp.Optional[str]:
    """Return the amount of time since the Sheep made some update
    to its log. Returns a str using the relevant time unit."""
    ping = None
    if sheep.log is not None and sheep.log.exists():
        delta = time.time() - sheep.log.stat().st_mtime
        if delta > 3600 * 24:
            ping = f'{delta / (3600 * 24):.1f}d'
        elif delta > 3600:
            ping = f'{delta / (3600):.1f}h'
        elif delta > 60:
            ping = f'{delta / 60:.1f}m'
        else:
            ping = f'{delta:.1f}s'
    return ping


class BaseExplorer(ABC, Explorer):
    """Base explorer for AudioCraft grids.

    All task specific solvers are expected to implement the `get_grid_metrics`
    method to specify logic about metrics to display for a given task.

    If additional stages are used, the child explorer must define how to handle
    these new stages in the `process_history` and `process_sheep` methods.
    """
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
            tt.leaf("sid", align="<"),
        ]

    @abstractmethod
    def get_grid_metrics(self):
        """Return the metrics that should be displayed in the tracking table.
        """
        ...

    def process_sheep(self, sheep, history):
        train = {
            "epoch": len(history),
        }
        parts = {"train": train}
        for metrics in history:
            for key, sub in metrics.items():
                part = parts.get(key, {})
                if 'duration' in sub:
                    # Convert to minutes for readability.
                    sub['duration'] = sub['duration'] / 60.
                part.update(sub)
                parts[key] = part
        ping = get_sheep_ping(sheep)
        if ping is not None:
            for name in self.stages():
                if name not in parts:
                    parts[name] = {}
                # Add the ping to each part for convenience.
                parts[name]['ping'] = ping
        return parts
