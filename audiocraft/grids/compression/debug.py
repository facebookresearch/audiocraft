# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Grid search file, simply list all the exp you want in `explorer`.
Any new exp added there will be scheduled.
You can cancel and experiment by commenting its line.

This grid is a minimal example for debugging compression task
and how to override parameters directly in a grid.
Learn more about dora grids: https://github.com/facebookresearch/dora
"""

from ._explorers import CompressionExplorer
from ...environment import AudioCraftEnvironment


@CompressionExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(gpus=2, partition=partitions)
    launcher.bind_(solver='compression/debug')

    with launcher.job_array():
        # base debug task using config from solver=compression/debug
        launcher()
        # we can override parameters in the grid to launch additional xps
        launcher({'rvq.bins': 2048, 'rvq.n_q': 4})
