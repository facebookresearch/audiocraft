# Env - chords_extraction on devfair

import pickle
import argparse
from chord_extractor.extractors import Chordino  # type: ignore
from chord_extractor import clear_conversion_cache, LabelledChordSequence  # type: ignore
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_jsonl_file', type=str, required=True,
                        help='abs path to .jsonl file containing list of absolute file paths seperated by new line')
    parser.add_argument('--target_output_dir', type=str, required=True,
                        help='target directory to save parsed chord files to, individual files will be saved inside')
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()
    return args


def save_to_db_cb(tgt_dir: str):
    # Every time one of the files has had chords extracted, receive the chords here
    # along with the name of the original file and then run some logic here, e.g. to
    # save the latest data to DB
    def inner(results: LabelledChordSequence):
        path = results.id.split(".wav")

        sequence = [(item.chord, item.timestamp) for item in results.sequence]

        if len(path) != 2:
            print("Something")
            print(path)
        else:
            file_idx = path[0].split("/")[-1]
            with open(f"{tgt_dir}/{file_idx}.chords", "wb") as f:
                # dump the object to the file
                pickle.dump(sequence, f)
    return inner


if __name__ == "__main__":
    '''This script extracts chord data from a list of audio files using the Chordino extractor,
    and saves the extracted chords to individual files in a target directory.'''
    print("parsed args")
    args = parse_args()
    files_to_extract_from = list()
    with open(args.src_jsonl_file, "r") as json_file:
        for line in tqdm(json_file.readlines()):
            # fpath = json.loads(line.replace("\n", ""))['path']
            fpath = line.replace("\n", "")
            if not args.override:
                fname = fpath.split("/")[-1].replace(".wav", ".chords")
                if os.path.exists(f"{args.target_output_dir}/{fname}"):
                    continue
            files_to_extract_from.append(line.replace("\n", ""))

    print(f"num files to parse: {len(files_to_extract_from)}")

    chordino = Chordino()

    # Optionally clear cache of file conversions (e.g. wav files that have been converted from midi)
    clear_conversion_cache()

    # Run bulk extraction
    res = chordino.extract_many(
        files_to_extract_from,
        callback=save_to_db_cb(args.target_output_dir),
        num_extractors=80,
        num_preprocessors=80,
        max_files_in_cache=400,
        stop_on_error=False,
    )
