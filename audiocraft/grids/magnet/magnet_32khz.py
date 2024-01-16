# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from ..musicgen._explorers import LMExplorer
from ...environment import AudioCraftEnvironment


@LMExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(gpus=32, partition=partitions)
    launcher.bind_(solver='magnet/magnet_32khz')
    # replace this by the desired music dataset
    launcher.bind_(dset='internal/music_400k_32khz')

    fsdp = {'autocast': False, 'fsdp.use': True}
    medium = {'model/lm/model_scale': 'medium'}
    adam = {'optim.optimizer': 'adamw', 'optim.lr': 1e-4}
    segdur_10secs = {'dataset.segment_duration': 10,
                     'dataset.batch_size': 576,
                     'generate.lm.decoding_steps': [20, 10, 10, 10]}

    # Small models (300M)
    launcher.slurm_(gpus=32).bind_(label='32gpus')
    with launcher.job_array():
        # 30 seconds
        sub = launcher.bind()
        sub()

        # 10 seconds
        sub = launcher.bind()
        sub(segdur_10secs)

    # Medium models (1.5B)
    launcher.bind_(fsdp)
    launcher.slurm_(gpus=64).bind_(label='64gpus')
    with launcher.job_array():
        # 30 seconds
        sub = launcher.bind()
        sub(medium, adam)

        # 10 seconds
        sub = launcher.bind()
        sub(segdur_10secs)
