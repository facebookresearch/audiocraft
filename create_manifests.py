#!/usr/bin/env python3
"""Create manifests for musiccaps_audio dataset."""
import argparse
from pathlib import Path
import random
from audiocraft.data.audio_dataset import find_audio_files, save_audio_meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory containing wav and json files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for manifests')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of train samples')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all wav files
    meta = find_audio_files(data_dir, exts=['.wav'], progress=True)

    # Shuffle
    random.seed(42)
    random.shuffle(meta)

    # Split
    n_train = int(len(meta) * args.train_ratio)
    train_meta = meta[:n_train]
    valid_meta = meta[n_train:]

    # Save
    save_audio_meta(output_dir / 'train.jsonl.gz', train_meta)
    save_audio_meta(output_dir / 'valid.jsonl.gz', valid_meta)

    print(f"Created train manifest with {len(train_meta)} samples")
    print(f"Created valid manifest with {len(valid_meta)} samples")

if __name__ == '__main__':
    main()