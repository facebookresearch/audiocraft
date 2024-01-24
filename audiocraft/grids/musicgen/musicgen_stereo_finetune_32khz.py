# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from ._explorers import LMExplorer
from ...environment import AudioCraftEnvironment


@LMExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(gpus=32, partition=partitions)
    launcher.bind_(solver='musicgen/musicgen_base_32khz')
    # replace this by the desired music dataset, which needs to be stereo
    launcher.bind_(dset='audio/example')

    fsdp = {'autocast': False, 'fsdp.use': True}
    medium = {'model/lm/model_scale': 'medium'}
    large = {'model/lm/model_scale': 'large'}

    cfg_low = {'classifier_free_guidance.training_dropout': 0.2}
    wd_low = {'conditioners.description.t5.word_dropout': 0.2}

    adam = {'optim.optimizer': 'adamw', 'optim.lr': 1e-4}

    stereo = {
        'codebooks_pattern.delay.delays': [0, 0, 1, 1, 2, 2, 3, 3],
        'transformer_lm.n_q': 8,
        'interleave_stereo_codebooks.use': True,
        'channels': 2,
    }

    # You must follow the instructions in docs/MUSICGEN.md about the creation
    # of the proper fine tuning checkpoints. We will assume they are stored under
    # ~/checkpoints/{mode_name}.

    checkpoints = Path.home() / 'checkpoints'

    launcher.bind_(fsdp, stereo, {'optim.epochs': 100})

    launcher.slurm_(gpus=32).bind_(label='32gpus')
    with launcher.job_array():
        sub = launcher.bind({'continue_from': str(checkpoints / 'stereo_finetune_musicgen-small.th')})
        sub()

    launcher.slurm_(gpus=64).bind_(label='64gpus')
    with launcher.job_array():
        sub = launcher.bind({'continue_from': str(checkpoints / 'stereo_finetune_musicgen-medium.th')})
        sub(medium, adam)

    launcher.slurm_(gpus=96).bind_(label='96gpus')
    with launcher.job_array():
        sub = launcher.bind({'continue_from': str(checkpoints / 'stereo_finetune_musicgen-large.th')})
        sub(large, cfg_low, wd_low, adam, {'optim.max_norm': 3})
