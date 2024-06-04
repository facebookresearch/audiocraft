# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
dora grid watermarking.kbits --clear
"""
import os
from audiocraft.environment import AudioCraftEnvironment
from ._explorers import WatermarkingMbExplorer


@WatermarkingMbExplorer
def explorer(launcher):
    partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])
    launcher.slurm_(
        gpus=8,
        partition=partitions,
        constraint="volta32gb",
    )
    launcher.bind_(
        {
            "solver": "watermark/robustness",
            "dset": os.getenv("AUDIOCRAFT_DSET", "audio/example"),
            "dataset.batch_size": 16,
            # optim
            "optim.epochs": 300,
            "schedule": {
                "lr_scheduler": "cosine",
                "cosine": {
                    "warmup": 4000,
                    "lr_min_ratio": 0.0,
                    "cycle_length": 1.0,
                },
            },
            # crop and padding
            "crop": {
                "prob": 0.4,
                "shuffle_prob": 0.2,
                "pad_prob": 0.2,
                "size": 0.5,
                "max_n_windows": 5,
            },
            # augmentations
            "select_aug_mode": 'use_eval',
            "aug_weights.updownresample": 0.1,
            "aug_weights.speed": 0.1,
            "aug_weights.echo": 0.1,
            "aug_weights.pink_noise": 0.1,
            "aug_weights.lowpass_filter": 0.1,
            "aug_weights.highpass_filter": 0.1,
            "aug_weights.bandpass_filter": 0.1,
            "aug_weights.smooth": 0.1,
            "aug_weights.boost_audio": 0.1,
            "aug_weights.duck_audio": 0.1,
            "aug_weights.mp3_compression": 0.1,
            "aug_weights.encodec": 0.1,
            "aug_weights.identity": 1.0,
            # multi-bit
            "audioseal.nbits": 16,
            "detector.output_dim": 32,
            "wm_mb.loss_type": "bce",
            "wm_mb.temperature": 0.1,
            # losses
            "losses": {  # encodec loss + tf  = 10
                "adv": 4.0,
                "feat": 4.0,
                "l1": 0.1,
                "mel": 0.0,
                "msspec": 2.0,
                "sisnr": 0.0,
                "tf_loudnessratio": 10.0,
            },
            "losses.wm_detection": 1.0,
            "losses.wm_mb": 1.0,
        }
    )
    launcher.bind_(label="kbits16")

    lrs = [5e-5]
    seeds = [1, 2, 3, 4]

    with launcher.job_array():
        for lr in lrs:
            for seed in seeds:
                launcher({
                    "optim.lr": lr,
                    "seed": seed,
                })
