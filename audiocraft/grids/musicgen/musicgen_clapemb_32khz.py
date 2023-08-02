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
    launcher.slurm_(gpus=32, partition=partitions)
    launcher.bind_(solver='musicgen/musicgen_base_32khz')
    # replace this by the desired music dataset
    launcher.bind_(dset='internal/music_400k_32khz')
    launcher.bind_(conditioner='clapemb2music')

    fsdp = {'autocast': False, 'fsdp.use': True}
    cache_path = {'conditioners.description.clap.cache_path':
                  '/fsx-audio-craft-llm/jadecopet/experiments/audiocraft/caches/clap_embed_music'}
    text_wav_training_opt = {'conditioners.description.clap.text_p': 0.5}

    launcher.bind_(fsdp)

    launcher.slurm_(gpus=32).bind_(label='32gpus')
    with launcher.job_array():
        launcher()
        launcher(text_wav_training_opt)
        launcher(cache_path)
        launcher(cache_path, text_wav_training_opt)
