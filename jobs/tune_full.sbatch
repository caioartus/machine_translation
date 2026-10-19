#!/bin/bash
# Learning-rate tuning at the full training size (15,362 examples), uniform sampling only, seed 0.
#
# This is a Slurm *job array*: one submission starts 3 independent tasks, one per learning rate:
#
#     task 0: lr = 3e-6     (the best value at size 1000)
#     task 1: lr = 1e-5
#     task 2: lr = 3e-5
#
# Why uniform only: at size 1000 (runs_tune2/) all three conditions chose the same
# learning rate, so tuning one condition is enough. The chosen learning rate is then
# used for every condition at this size.
#
# Why these learning rates: at size 1000 the problem was overfitting, which favours
# small learning rates. With 15x more data the risk is undertraining instead, so the
# grid goes upward from 3e-6. If full also picks 3e-6, the same value is used for
# 2000 and 5000 without tuning them; if it picks a larger one, 5000 is tuned too.
#
# Budget, fixed in advance (not read off the dev curve): 2900 steps x batch 16
# = about 3 epochs of the full data. Warm-up is 10% of the steps.
#
# The run with the chosen learning rate also counts as the final uniform seed-0 run,
# so checkpoints are kept: the winner's is needed later for test evaluation. Delete
# the other two once the choice is made (each is ~2 GB).
#
# Run from the repository root:
#     sbatch jobs/tune_full.sbatch
# Re-run only some tasks (e.g. after a failure):
#     sbatch --array=1 jobs/tune_full.sbatch
# The tasks run 2 at a time, because the cluster allows 2 GPUs per user (QOSMaxGRESPerUser).
#
# Outputs (one folder per learning rate, so runs never overwrite each other):
#     runs_tune_full/lr<LR>/uniform_nfull_seed0/{best/, dev_curve.csv, run_info.json}
# Logs:
#     logs/slurm/mt-tune-full-<array job id>_<task id>.out
#
# Based on job_template.sbatch from the Unofficial SoC Cluster Guide.

#SBATCH --job-name=mt-tune-full
#SBATCH --array=0-2%2          # %2: at most 2 tasks run at once (our GPU limit per user)
#SBATCH --partition=gpu-long
#SBATCH --time=04:00:00        # generous: ~10x the steps of round 2, plus more evaluations
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000

set -euo pipefail

# --- What to tune ----------------------------------------------------------------
LEARNING_RATES=(3e-6 1e-5 3e-5)
CONDITION=uniform
SIZE=full
SEED=0
OUT_ROOT=runs_tune_full

# Same for every task.
STEPS=2900            # about 3 epochs of 15,362 examples at batch size 16
EVAL_EVERY=100        # 29 evaluations along the run
WARMUP_STEPS=290      # 10% of the steps

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

# --- Pick this task's learning rate ----------------------------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LR="${LEARNING_RATES[$TASK_ID]}"

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
    --steps "$STEPS" \
    --eval_every "$EVAL_EVERY" \
    --warmup_steps "$WARMUP_STEPS" \
    --out_dir "$OUT_DIR"

echo "Finished $(date)"
