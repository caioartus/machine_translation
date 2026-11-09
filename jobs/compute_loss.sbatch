#!/bin/bash
# Score every training pair with the pretrained M2M-100 (loss-based difficulty).
# Run from the repository root:  sbatch jobs/compute_loss.sbatch [compute_loss.py arguments]
#   sbatch jobs/compute_loss.sbatch                       # ALMA train
#   sbatch jobs/compute_loss.sbatch --dataset alma_news   # ALMA + News Commentary
#   sbatch jobs/compute_loss.sbatch --limit 200           # smoke test
# Output: data/features/<stem>_loss.csv, log in logs/slurm/.

#SBATCH --job-name=mt-loss
#SBATCH --partition=gpu-long
#SBATCH --time=00:30:00
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mt}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-$$}.out" 2>&1
cd "$REPO_ROOT"

set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

echo "Job ${SLURM_JOB_ID:-local} on $(hostname), started $(date)"
echo "compute_loss.py arguments: $*"
nvidia-smi

python -u scripts/compute_loss.py "$@"

echo "Finished $(date)"
