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
    launcher.slurm_(gpus=64, partition=partitions)
    launcher.bind_(solver='audiogen/audiogen_base_16khz')
    # replace this by the desired environmental sound dataset
    launcher.bind_(dset='internal/sounds_16khz')

    fsdp = {'autocast': False, 'fsdp.use': True}
    medium = {'model/lm/model_scale': 'medium'}

    launcher.bind_(fsdp)
    launcher(medium)
