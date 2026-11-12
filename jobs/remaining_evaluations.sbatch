#!/bin/bash
#SBATCH --job-name=mt-remaining
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/mt-remaining-%A_%a.out

# Usage (after `python scripts/run_remaining_evaluations.py plan --runs_dir ...`):
#   sbatch --array=$(cat results/_evaluation_workflow/pending.txt) jobs/remaining_evaluations.sbatch
# With a custom workflow folder, pass it to both: ... jobs/remaining_evaluations.sbatch --work_dir <dir>
# COMET_PYTHON and HF_HOME default to Jake's shared COMET setup; export your own to override.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
umask 022

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mt

export COMET_PYTHON="${COMET_PYTHON:-/mnt/scratch/t/teng/comet-work/venv/bin/python}"
export HF_HOME="${HF_HOME:-/mnt/scratch/t/teng/comet-work/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export TMPDIR
TMPDIR=$(mktemp -d "/tmp/mt-remaining-${SLURM_JOB_ID}-XXXXXX")
trap 'rmdir "$TMPDIR" 2>/dev/null || true' EXIT

python -u scripts/run_remaining_evaluations.py task "$SLURM_ARRAY_TASK_ID" "$@"
