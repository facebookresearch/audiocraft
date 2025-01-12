#!/bin/zsh
#SBATCH --job-name=my_job_array
#SBATCH --array=0-N  # adjust the range of indices as needed
#SBATCH --output=logs/%A_%a.out  # output file name format, this assumes there exists a <root>/logs directory
#SBATCH --error=logs/%A_%a.err  # error file name format, this assumes there exists a <root>/logs directory
#SBATCH --time=01:00:00  # adjust the time limit as needed
#SBATCH --nodes=1  # adjust the number of nodes as needed
#SBATCH --ntasks-per-node=1  # adjust the number of tasks per node as needed
#SBATCH --cpus-per-task=8  # adjust the number of CPUs per task as needed
#SBATCH --mem-per-cpu=16G  # adjust the memory per CPU as needed

# Load any necessary modules or dependencies
conda activate your_env

# run extraction of chords in job array
python scripts/chords/extract_chords.py --src_jsonl_file /path/to/parsed/filepaths_${SLURM_ARRAY_TASK_ID}.jsonl --target_output_dir /target/directory/to/save/chords/to --path_to_pre_defined_map /path/to/predefined/chord_to_index_mapping.pkl

