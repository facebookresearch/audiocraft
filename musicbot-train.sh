#!/bin/bash

# Check if dataset argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path-to-dataset>"
    exit 1
fi

# Interface: musicbot-train --model-name="grimes" --path-to-dataset="gs://FOLDER"
git clone https://github.com/createsafe/audiocraft && cd audiocraft
python -m pip install 'torch==2.1.0'
pip install -e .

# Grab UUID folder with *.json and *.mp3 pairs
gsutil cp -r gs://musicgen-datasets/"$1" .
mv "$1" dataset
mkdir dataset/train
mkdir dataset/val
mkdir dataset/eval
mkdir dataset/gen

# Train/Val/Eval/Gen split
cd utils && python split.py
cd ..
python -m audiocraft.data.audio_dataset dataset/train manifests/train/data.jsonl
python -m audiocraft.data.audio_dataset dataset/val manifests/val/data.jsonl
python -m audiocraft.data.audio_dataset dataset/eval manifests/eval/data.jsonl
python -m audiocraft.data.audio_dataset dataset/gen manifests/gen/data.jsonl

# Fine tune the model 
dora run -d solver=musicgen/musicgen_base_32khz model/lm/model_scale=small \
    continue_from=//pretrained/facebook/musicgen-small conditioner=text2music \
    dset=audio/data

