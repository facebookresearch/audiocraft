# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluation with objective metrics for the pretrained audio-MAGNeT models.
This grid takes signature from the training grid and runs evaluation-only stage.

When running the grid for the first time, please use:
REGEN=1 dora grid magnet.audio_magnet_pretrained_16khz_eval
and re-use the REGEN=1 option when the grid is changed to force regenerating it.

Note that you need the proper metrics external libraries setup to use all
the objective metrics activated in this grid. Refer to the README for more information.
"""

import os

from ..musicgen._explorers import GenerationEvalExplorer
from ...environment import AudioCraftEnvironment
from ... import train


def eval(launcher, batch_size: int = 32):
    opts = {
        'dset': 'audio/audiocaps_16khz',
        'solver/audiogen/evaluation': 'objective_eval',
        'execute_only': 'evaluate',
        '+dataset.evaluate.batch_size': batch_size,
        '+metrics.fad.tf.batch_size': 32,
    }
    # binary for FAD computation: replace this path with your own path
    metrics_opts = {
        'metrics.fad.tf.bin': '/data/home/jadecopet/local/usr/opt/google-research'
    }

    sub = launcher.bind(opts)
    sub.bind_(metrics_opts)

    # base objective metrics
    sub()


@GenerationEvalExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(gpus=4, partition=partitions)

    if 'REGEN' not in os.environ:
        folder = train.main.dora.dir / 'grids' / __name__.split('.', 2)[-1]
        with launcher.job_array():
            for sig in folder.iterdir():
                if not sig.is_symlink():
                    continue
                xp = train.main.get_xp_from_sig(sig.name)
                launcher(xp.argv)
        return

    with launcher.job_array():
        audio_magnet = launcher.bind(solver="magnet/audio_magnet_16khz")

        fsdp = {'autocast': False, 'fsdp.use': True}

        # Small audio-MAGNeT model (300M)
        audio_magnet_small = audio_magnet.bind({'continue_from': '//pretrained/facebook/audio-magnet-small'})
        eval(audio_magnet_small, batch_size=128)

        # Medium audio-MAGNeT model (1.5B)
        audio_magnet_medium = audio_magnet.bind({'continue_from': '//pretrained/facebook/audio-magnet-medium'})
        audio_magnet_medium.bind_({'model/lm/model_scale': 'medium'})
        audio_magnet_medium.bind_(fsdp)
        eval(audio_magnet_medium, batch_size=128)
