# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from ._explorers import LMExplorer
from ...environment import AudioCraftEnvironment


@LMExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(gpus=64, partition=partitions, constraint='volta32gb').bind_(label='64gpus')
    launcher.bind_(dset='internal/music_400k_32khz')

    sub = launcher.bind_({'solver': 'musicgen/musicgen_style_32khz',
                          'autocast': False,
                          'fsdp.use': True,
                          'model/lm/model_scale': 'medium',
                          'optim.optimizer': 'adamw',
                          'optim.lr': 1e-4,
                          'generate.every': 25,
                          'dataset.generate.num_samples': 64,
                          })
    sub()
