# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluation with objective metrics for the pretrained AudioGen models.
This grid takes signature from the training grid and runs evaluation-only stage.

When running the grid for the first time, please use:
REGEN=1 dora grid audiogen.audiogen_pretrained_16khz_eval
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
    opt1 = {'generate.lm.use_sampling': True, 'generate.lm.top_k': 250, 'generate.lm.top_p': 0.}
    opt2 = {'transformer_lm.two_step_cfg': True}

    sub = launcher.bind(opts)
    sub.bind_(metrics_opts)

    # base objective metrics
    sub(opt1, opt2)


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

    audiogen_base = launcher.bind(solver="audiogen/audiogen_base_16khz")
    audiogen_base.bind_({'autocast': False, 'fsdp.use': True})

    audiogen_base_medium = audiogen_base.bind({'continue_from': '//pretrained/facebook/audiogen-medium'})
    audiogen_base_medium.bind_({'model/lm/model_scale': 'medium'})
    eval(audiogen_base_medium, batch_size=128)
