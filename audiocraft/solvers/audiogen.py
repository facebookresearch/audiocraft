# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from . import builders, musicgen


class AudioGenSolver(musicgen.MusicGenSolver):
    """Solver for AudioGen re-implementation training task.

    Note that this implementation does not strictly follows
    the method proposed in https://arxiv.org/abs/2209.15352
    but is derived from MusicGen's training pipeline.

    More information can be found in the AudioGen model card.
    """
    DATASET_TYPE: builders.DatasetType = builders.DatasetType.SOUND
