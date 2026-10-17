#!/bin/bash
# Run from the repository root:  sbatch job.sbatch [train.py arguments]
# Arguments are forwarded to scripts/train.py, e.g.
#   sbatch job.sbatch --condition joint --size 1000 --seed 0
# With no arguments, a quick smoke test is run (20 steps).
# Override resources on the command line without editing this file, e.g.
#   sbatch --time=02:00:00 job.sbatch --condition uniform --size 5000 --seed 0
# Based on job_template.sbatch from the Unofficial SoC Cluster Guide.

#SBATCH --job-name=mt-train
#SBATCH --partition=gpu-long
#SBATCH --time=00:30:00
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000

set -euo pipefail

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mt}"

# Slurm runs a copy of this script, so use the directory sbatch was called from.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-$$}.out" 2>&1
cd "$REPO_ROOT"

# Non-interactive jobs don't source ~/.bashrc, so initialise conda explicitly.
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

# No arguments given: run the smoke test.
if [ "$#" -eq 0 ]; then
    set -- --condition joint --size 1000 --seed 0 --steps 20 --eval_every 10 --out_dir runs_test
fi

echo "Job ${SLURM_JOB_ID:-local} on $(hostname), started $(date)"
echo "train.py arguments: $*"
nvidia-smi

# -u: unbuffered output, so the log can be followed live with tail -f.
python -u scripts/train.py "$@"

echo "Finished $(date)"
