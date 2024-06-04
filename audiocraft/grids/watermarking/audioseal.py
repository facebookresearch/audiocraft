# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
dora grid watermarking.audioseal --clear
"""
from audiocraft.environment import AudioCraftEnvironment
from ._explorers import WatermarkingExplorer


@WatermarkingExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(
        gpus=8,
        partition=partitions,
        constraint="volta32gb",
    )
    launcher.bind_(
        {
            "solver": "watermark/robustness",
            "dset": "audio/example",
        }
    )
    launcher.bind_(label="audioseal")

    with launcher.job_array():
        launcher()
