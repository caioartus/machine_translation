#!/bin/bash
#SBATCH --job-name=mt-remaining
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/mt-remaining-%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
umask 022

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mt

export COMET_PYTHON=/mnt/scratch/t/teng/comet-work/venv/bin/python
export HF_HOME=/mnt/scratch/t/teng/comet-work/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export TMPDIR
TMPDIR=$(mktemp -d "/tmp/mt-remaining-${SLURM_JOB_ID}-XXXXXX")
trap 'rmdir "$TMPDIR" 2>/dev/null || true' EXIT

python -u scripts/run_remaining_evaluations.py task "$SLURM_ARRAY_TASK_ID"
