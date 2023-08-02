# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Grid search file, simply list all the exp you want in `explorer`.
Any new exp added there will be scheduled.
You can cancel and experiment by commenting its line.

This grid shows how to train the new AudioGen EnCodec model at 16 kHz.
"""

from ._explorers import CompressionExplorer
from ...environment import AudioCraftEnvironment


@CompressionExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(gpus=8, partition=partitions)
    # use configuration for AudioGen's EnCodec model trained on monophonic audio sampled at 16 kHz
    # AudioGen's EnCodec is trained with a total stride of 320 leading to a frame rate of 50 hz
    launcher.bind_(solver='compression/encodec_audiogen_16khz')
    # replace this by the desired sound dataset
    launcher.bind_(dset='internal/sounds_16khz')
    # launch xp
    launcher()
