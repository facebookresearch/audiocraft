import os
import shutil
import random
from pathlib import Path

# Set the source directory containing the mp3/json pairs
source_directory = "../dataset"

# Set the destination directories
dest_directories = {
    "train": "../dataset/train/",
    "val": "../dataset/val",
    "eval": "../dataset/eval",
    "gen": "../dataset/gen"
}

# Define the split ratios
split_ratios = {
    "train": 0.8,
    "val": 0.1,
    "eval": 0.05,
    "gen": 0.05
}

# Create destination directories if they don't exist
for directory in dest_directories.values():
    os.makedirs(directory, exist_ok=True)

# Get list of mp3/json pairs
file_pairs = [(f.stem, f) for f in Path(source_directory).glob("*.mp3")]

# Shuffle the file pairs
random.shuffle(file_pairs)

# Calculate split indices
total_files = len(file_pairs)
split_indices = {
    "train": int(total_files * split_ratios["train"]),
    "val": int(total_files * (split_ratios["train"] + split_ratios["val"])),
    "eval": int(total_files * (split_ratios["train"] + split_ratios["val"] + split_ratios["eval"]))
}

# Move files to respective directories
for idx, (uuid, file) in enumerate(file_pairs):
    if idx < split_indices["train"]:
        dest_dir = dest_directories["train"]
    elif idx < split_indices["val"]:
        dest_dir = dest_directories["val"]
    elif idx < split_indices["eval"]:
        dest_dir = dest_directories["eval"]
    else:
        dest_dir = dest_directories["gen"]
    
    # Move both mp3 and json files
    shutil.move(file, os.path.join(dest_dir, f"{uuid}.mp3"))
    shutil.move(os.path.join(source_directory, f"{uuid}.json"), os.path.join(dest_dir, f"{uuid}.json"))

print("Files moved successfully!")

