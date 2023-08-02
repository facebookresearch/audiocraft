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

    fsdp = {'autocast': False, 'fsdp.use': True}
    medium = {'model/lm/model_scale': 'medium'}
    large = {'model/lm/model_scale': 'large'}

    cfg_low = {'classifier_free_guidance.training_dropout': 0.2}
    wd_low = {'conditioners.description.t5.word_dropout': 0.2}

    adam = {'optim.optimizer': 'adamw', 'optim.lr': 1e-4}

    # BEGINNING OF CACHE WRITING JOBS.
    cache_write = {
        'cache.path': '/fsx-codegen/defossez/cache/interleave_stereo_nv_32k',
        'cache.write': True,
        'generate.every': 500,
        'evaluate.every': 500,
        'logging.log_updates': 50,
    }

    cache_sub = launcher.bind({'model/lm/model_scale': 'xsmall', 'conditioner': 'none'})
    cache_sub.bind_({'deadlock.use': True})
    cache_sub.slurm_(gpus=8)
    with launcher.job_array():
        num_shards = 10  # total number of jobs running in parallel.
        for shard in range(0, num_shards):
            launcher(cache_write, {'cache.write_num_shards': num_shards, 'cache.write_shard': shard})

    # REMOVE THE FOLLOWING RETURN STATEMENT ONCE THE ABOVE JOBS ARE DONE,
    # OR SUFFICIENTLY AHEAD.
    return

    cache = {
        'cache.path': '/fsx-codegen/defossez/cache/interleave_stereo_nv_32k',
    }
    launcher.bind_(fsdp, cache)

    launcher.slurm_(gpus=32).bind_(label='32gpus')
    with launcher.job_array():
        sub = launcher.bind()
        sub()

    launcher.slurm_(gpus=64).bind_(label='64gpus')
    with launcher.job_array():
        sub = launcher.bind()
        sub(medium, adam)

    launcher.slurm_(gpus=96).bind_(label='96gpus')
    with launcher.job_array():
        sub = launcher.bind()
        sub(large, cfg_low, wd_low, adam, {'optim.max_norm': 3})
