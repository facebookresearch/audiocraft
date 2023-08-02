# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Resampling script.
"""
import argparse
from pathlib import Path
import shutil
import typing as tp

import submitit
import tqdm

from audiocraft.data.audio import audio_read, audio_write
from audiocraft.data.audio_dataset import load_audio_meta, find_audio_files
from audiocraft.data.audio_utils import convert_audio
from audiocraft.environment import AudioCraftEnvironment


def read_txt_files(path: tp.Union[str, Path]):
    with open(args.files_path) as f:
        lines = [line.rstrip() for line in f]
        print(f"Read {len(lines)} in .txt")
        lines = [line for line in lines if Path(line).suffix not in ['.json', '.txt', '.csv']]
        print(f"Filtered and keep {len(lines)} from .txt")
        return lines


def read_egs_files(path: tp.Union[str, Path]):
    path = Path(path)
    if path.is_dir():
        if (path / 'data.jsonl').exists():
            path = path / 'data.jsonl'
        elif (path / 'data.jsonl.gz').exists():
            path = path / 'data.jsonl.gz'
        else:
            raise ValueError("Don't know where to read metadata from in the dir. "
                             "Expecting either a data.jsonl or data.jsonl.gz file but none found.")
    meta = load_audio_meta(path)
    return [m.path for m in meta]


def process_dataset(args, n_shards: int, node_index: int, task_index: tp.Optional[int] = None):
    if task_index is None:
        env = submitit.JobEnvironment()
        task_index = env.global_rank
    shard_index = node_index * args.tasks_per_node + task_index

    if args.files_path is None:
        lines = [m.path for m in find_audio_files(args.root_path, resolve=False, progress=True, workers=8)]
    else:
        files_path = Path(args.files_path)
        if files_path.suffix == '.txt':
            print(f"Reading file list from .txt file: {args.files_path}")
            lines = read_txt_files(args.files_path)
        else:
            print(f"Reading file list from egs: {args.files_path}")
            lines = read_egs_files(args.files_path)

    total_files = len(lines)
    print(
        f"Total of {total_files} processed with {n_shards} shards. " +
        f"Current idx = {shard_index} -> {total_files // n_shards} files to process"
    )
    for idx, line in tqdm.tqdm(enumerate(lines)):

        # skip if not part of this shard
        if idx % n_shards != shard_index:
            continue

        path = str(AudioCraftEnvironment.apply_dataset_mappers(line))
        root_path = str(args.root_path)
        if not root_path.endswith('/'):
            root_path += '/'
        assert path.startswith(str(root_path)), \
            f"Mismatch between path and provided root: {path} VS {root_path}"

        try:
            metadata_path = Path(path).with_suffix('.json')
            out_path = args.out_path / path[len(root_path):]
            out_metadata_path = out_path.with_suffix('.json')
            out_done_token = out_path.with_suffix('.done')

            # don't reprocess existing files
            if out_done_token.exists():
                continue

            print(idx, out_path, path)
            mix, sr = audio_read(path)
            mix_channels = args.channels if args.channels is not None and args.channels > 0 else mix.size(0)
            # enforce simple stereo
            out_channels = mix_channels
            if out_channels > 2:
                print(f"Mix has more than two channels: {out_channels}, enforcing 2 channels")
                out_channels = 2
            out_sr = args.sample_rate if args.sample_rate is not None else sr
            out_wav = convert_audio(mix, sr, out_sr, out_channels)
            audio_write(out_path.with_suffix(''), out_wav, sample_rate=out_sr,
                        format=args.format, normalize=False, strategy='clip')
            if metadata_path.exists():
                shutil.copy(metadata_path, out_metadata_path)
            else:
                print(f"No metadata found at {str(metadata_path)}")
            out_done_token.touch()
        except Exception as e:
            print(f"Error processing file line: {line}, {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Resample dataset with SLURM.")
    parser.add_argument(
        "--log_root",
        type=Path,
        default=Path.home() / 'tmp' / 'resample_logs',
    )
    parser.add_argument(
        "--files_path",
        type=Path,
        help="List of files to process, either .txt (one file per line) or a jsonl[.gz].",
    )
    parser.add_argument(
        "--root_path",
        type=Path,
        required=True,
        help="When rewriting paths, this will be the prefix to remove.",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        required=True,
        help="When rewriting paths, `root_path` will be replaced by this.",
    )
    parser.add_argument("--xp_name", type=str, default="shutterstock")
    parser.add_argument(
        "--nodes",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--tasks_per_node",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--memory_gb",
        type=int,
        help="Memory in GB."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="wav",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=32000,
    )
    parser.add_argument(
        "--channels",
        type=int,
    )
    parser.add_argument(
        "--partition",
        default='learnfair',
    )
    parser.add_argument("--qos")
    parser.add_argument("--account")
    parser.add_argument("--timeout", type=int, default=4320)
    parser.add_argument('--debug', action='store_true', help='debug mode (local run)')
    args = parser.parse_args()
    n_shards = args.tasks_per_node * args.nodes
    if args.files_path is None:
        print("Warning: --files_path not provided, not recommended when processing more than 10k files.")
    if args.debug:
        print("Debugging mode")
        process_dataset(args, n_shards=n_shards, node_index=0, task_index=0)
    else:

        log_folder = Path(args.log_root) / args.xp_name / '%j'
        print(f"Logging to: {log_folder}")
        log_folder.parent.mkdir(parents=True, exist_ok=True)
        executor = submitit.AutoExecutor(folder=str(log_folder))
        if args.qos:
            executor.update_parameters(slurm_partition=args.partition, slurm_qos=args.qos, slurm_account=args.account)
        else:
            executor.update_parameters(slurm_partition=args.partition)
        executor.update_parameters(
            slurm_job_name=args.xp_name, timeout_min=args.timeout,
            cpus_per_task=args.cpus_per_task, tasks_per_node=args.tasks_per_node, nodes=1)
        if args.memory_gb:
            executor.update_parameters(mem=f'{args.memory_gb}GB')
        jobs = []
        with executor.batch():
            for node_index in range(args.nodes):
                job = executor.submit(process_dataset, args, n_shards=n_shards, node_index=node_index)
                jobs.append(job)
        for job in jobs:
            print(f"Waiting on job {job.job_id}")
            job.results()
