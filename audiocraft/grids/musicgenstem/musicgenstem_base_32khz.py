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
    launcher.slurm_(gpus=64, partition=partitions).bind_(label='64gpus')

    # replace this by the desired music dataset
    launcher.bind_(dset='internal/music_400k_32khz')

    sub = launcher.bind_({'solver': 'musicgen_sourcesep/musicgen_sourcesep_base_32khz',
                          'autocast': False, 
                          'fsdp.use': True,
                          'model/lm/model_scale': 'medium',
                          'optim.optimizer': 'adamw', 
                          'optim.lr': 1e-4, 
                          'generate.every': 25,
                          'dataset.generate.num_samples': 64,
                          })
    
    sub = launcher.bind()

    sub({'transformer_lm.n_q': 6, 
         'codebooks_pattern.delay.delays': [0, 0, 0, 1, 2, 3],})


    sub({'transformer_lm.n_q': 7, 
         'codebooks_pattern.delay.delays': [0, 1, 0, 0, 1, 2, 3],
         'multistem_compression_model_checkpoints.pretrained': 'facebook/encodec_32_khz_bass_2_drums_1_other_4'
         })
