# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import pickle
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chords_folder', type=str, required=True,
                        help='path to directory containing parsed chords files')
    parser.add_argument('--output_directory', type=str, required=False,
                        help='path to output directory to generate code maps to, \
                            if not given - chords_folder would be used', default='')
    parser.add_argument('--path_to_pre_defined_map', type=str, required=False,
                        help='for evaluation purpose, use pre-defined chord-to-index map', default='')
    args = parser.parse_args()
    return args


def get_chord_dict(chord_folder: str):
    chord_dict = {}
    distinct_chords = set()

    chord_to_index = {}  # Mapping between chord and index
    index_counter = 0

    for filename in tqdm(os.listdir(chord_folder)):
        if filename.endswith(".chords"):
            idx = filename.split(".")[0]

            with open(os.path.join(chord_folder, filename), "rb") as file:
                chord_data = pickle.load(file)

            for chord, _ in chord_data:
                distinct_chords.add(chord)
                if chord not in chord_to_index:
                    chord_to_index[chord] = index_counter
                    index_counter += 1

            chord_dict[idx] = chord_data
    chord_to_index["UNK"] = index_counter
    return chord_dict, distinct_chords, chord_to_index


def get_predefined_chord_to_index_map(path_to_chords_to_index_map: str):
    def inner(chord_folder: str):
        chords_to_index = pickle.load(open(path_to_chords_to_index_map, "rb"))
        distinct_chords = set(chords_to_index.keys())
        chord_dict = {}
        for filename in tqdm(os.listdir(chord_folder), desc=f'iterating: {chord_folder}'):
            if filename.endswith(".chords"):
                idx = filename.split(".")[0]

                with open(os.path.join(chord_folder, filename), "rb") as file:
                    chord_data = pickle.load(file)

                chord_dict[idx] = chord_data
        return chord_dict, distinct_chords, chords_to_index
    return inner


if __name__ == "__main__":
    '''This script processes and maps chord data from a directory of parsed chords files,
    generating two output files: a combined chord dictionary and a chord-to-index mapping.'''
    args = parse_args()
    chord_folder = args.chords_folder
    output_dir = args.output_directory
    if output_dir == '':
        output_dir = chord_folder
    func = get_chord_dict
    if args.path_to_pre_defined_map != "":
        func = get_predefined_chord_to_index_map(args.path_to_pre_defined_map)

    chord_dict, distinct_chords, chord_to_index = func(chord_folder)

    # Save the combined chord dictionary as a pickle file
    combined_filename = os.path.join(output_dir, "combined_chord_dict.pkl")
    with open(combined_filename, "wb") as file:
        pickle.dump(chord_dict, file)

    # Save the chord-to-index mapping as a pickle file
    mapping_filename = os.path.join(output_dir, "chord_to_index_mapping.pkl")
    with open(mapping_filename, "wb") as file:
        pickle.dump(chord_to_index, file)

    print("Number of distinct chords:", len(distinct_chords))
    print("Chord dictionary:", chord_to_index)
