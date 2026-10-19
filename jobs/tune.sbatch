#!/bin/bash
# Learning-rate tuning for all three sampling conditions, at size 1000, seed 0.
#
# This is a Slurm *job array*: one submission starts 9 independent tasks,
# one per (learning rate, condition) pair:
#
#                  uniform   length   joint
#     lr = 1e-5       0         1        2
#     lr = 3e-5       3         4        5
#     lr = 1e-4       6         7        8
#
# Every other setting (steps, batch size, warm-up, weight decay, cap) is left at the
# train.py default, so the only things that vary are the learning rate and the condition.
#
# Run from the repository root:
#     sbatch jobs/tune.sbatch
# Re-run only some tasks (e.g. after a failure):
#     sbatch --array=4,7 jobs/tune.sbatch
#
# Outputs (one folder per learning rate, so runs never overwrite each other):
#     runs_tune/lr<LR>/<condition>_n1000_seed0/{dev_curve.csv, run_info.json}
# Logs:
#     logs/slurm/mt-tune-<array job id>_<task id>.out
#
# Based on job_template.sbatch from the Unofficial SoC Cluster Guide.

#SBATCH --job-name=mt-tune
#SBATCH --array=0-8
#SBATCH --partition=gpu-long
#SBATCH --time=01:00:00
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000

set -euo pipefail

# --- What to tune ----------------------------------------------------------------
LEARNING_RATES=(1e-5 3e-5 1e-4)
CONDITIONS=(uniform length joint)
SIZE=1000
SEED=0
OUT_ROOT=runs_tune

# The saved checkpoint (~2 GB) is only needed to compute dev BLEU, which train.py
# already does before it exits. Set to 1 to keep the checkpoints anyway.
KEEP_CHECKPOINTS=0

# --- Environment -----------------------------------------------------------------
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mt}"

# Slurm runs a copy of this script, so use the directory sbatch was called from.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/${SLURM_JOB_NAME:-job}-${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.out" 2>&1
cd "$REPO_ROOT"

# Non-interactive jobs don't source ~/.bashrc, so initialise conda explicitly.
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

# --- Pick this task's (learning rate, condition) pair ----------------------------
# Task id -> row (learning rate) and column (condition) of the table above.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_CONDITIONS=${#CONDITIONS[@]}
LR="${LEARNING_RATES[$((TASK_ID / N_CONDITIONS))]}"
CONDITION="${CONDITIONS[$((TASK_ID % N_CONDITIONS))]}"

OUT_DIR="$OUT_ROOT/lr$LR"
RUN_DIR="$OUT_DIR/${CONDITION}_n${SIZE}_seed${SEED}"   # the folder train.py writes to

echo "Job ${SLURM_ARRAY_JOB_ID:-local}, task $TASK_ID on $(hostname), started $(date)"
echo "condition: $CONDITION, lr: $LR, size: $SIZE, seed: $SEED -> $RUN_DIR"
nvidia-smi

# --- Train -----------------------------------------------------------------------
# -u: unbuffered output, so the log can be followed live with tail -f.
python -u scripts/train.py \
    --condition "$CONDITION" \
    --size "$SIZE" \
    --seed "$SEED" \
    --lr "$LR" \
    --out_dir "$OUT_DIR"

# --- Clean up --------------------------------------------------------------------
# Only reached if training succeeded (set -e stops the script on any error).
if [ "$KEEP_CHECKPOINTS" -eq 0 ]; then
    rm -rf "$RUN_DIR/best"
    echo "Deleted checkpoint $RUN_DIR/best (KEEP_CHECKPOINTS=0)"
fi

echo "Finished $(date)"
