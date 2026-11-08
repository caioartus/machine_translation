#!/bin/bash
# Submit from repository root; override partition/GPU to match your working job.
#SBATCH --job-name=mt-evaluate
#SBATCH --partition=gpu-long
#SBATCH --time=01:00:00
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000

set -euo pipefail
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mt}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$REPO_ROOT/logs/slurm"
exec > "$REPO_ROOT/logs/slurm/${SLURM_JOB_NAME:-mt-evaluate}-${SLURM_JOB_ID:-$$}.out" 2>&1
cd "$REPO_ROOT"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u
nvidia-smi
echo "Started $(date) on $(hostname)"
python -u scripts/evaluate.py --device cuda "$@"
echo "Finished $(date)"
