#!/bin/bash

export USER=sang
# CHANGE THIS

command="dora -P audiocraft run \
    solver=musicgen/musicgen_melody_32khz \
    model/lm/model_scale=medium \
    continue_from=//pretrained/facebook/musicgen-melody \
    conditioner=chroma2music \
    dset=audio/neto_refine \
    dataset.num_workers=2 \
    dataset.valid.num_samples=1 \
    dataset.batch_size=2 \
    schedule.cosine.warmup=8 \
    optim.optimizer=adamw \
    optim.lr=1e-4 \
    optim.epochs=50 \
    optim.updates_per_epoch=2000 \
    optim.adam.weight_decay=0.01 \
    generate.lm.prompted_samples=False \
    generate.lm.gen_gt_samples=True"

# CHANGE THIS (dataset.batch_size)
# uses dadaw by default, which is worse for single-gpu runs
# stops training after 5 epochs- change this
# 2000 by default, change this if you want checkpoints quicker ig
# skip super long generate step

$command