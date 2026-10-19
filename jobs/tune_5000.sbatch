#!/bin/bash
# Learning-rate tuning at size 5000, uniform sampling only, seed 0.
#
# This is a Slurm *job array*: one submission starts one task per learning rate:
#
#     task 0: lr = 3e-6     (the best value at size 1000)
#     task 1: lr = 1e-5     (the best value at full size)
#
# Why: the best learning rate depends on size (3e-6 at 1000, 1e-5 at full), so 5000 is
# tuned too. The two previous winners bracket it, so they are the grid.
#
# Why uniform only: at size 1000 (runs_tune2/) all three conditions chose the same
# learning rate, so tuning one condition is enough. The chosen learning rate is then
# used for every condition at this size.
#
# Budget, fixed in advance (same rule as full): 940 steps x batch 16 = about 3 epochs
# of 5000 examples. Warm-up is 10% of the steps.
#
# Decision rule, fixed before the results: pick the learning rate with the lower dev loss
# (checkpoint selection uses dev loss too). If the two are within about 0.01, either is fine.
#
# The run with the chosen learning rate also counts as the final uniform seed-0 run,
# so checkpoints are kept: the winner's is needed later for test evaluation. Delete
# the other once the choice is made (~2 GB).
#
# Run from the repository root:
#     sbatch jobs/tune_5000.sbatch
# Re-run only some tasks (e.g. after a failure):
#     sbatch --array=1 jobs/tune_5000.sbatch
# Both tasks run at once: the cluster allows 2 GPUs per user (QOSMaxGRESPerUser).
#
# Outputs (one folder per learning rate, so runs never overwrite each other):
#     runs_tune_5000/lr<LR>/uniform_n5000_seed0/{best/, dev_curve.csv, run_info.json}
# Logs:
#     logs/slurm/mt-tune-5000-<array job id>_<task id>.out
#
# Based on job_template.sbatch from the Unofficial SoC Cluster Guide.

#SBATCH --job-name=mt-tune-5000
#SBATCH --array=0-1            # 2 tasks, so both run at once on our 2 GPUs
#SBATCH --partition=gpu-long
#SBATCH --time=02:00:00        # generous: a third of the full-size steps, which fit in 4 h
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000

set -euo pipefail

# --- What to tune ----------------------------------------------------------------
LEARNING_RATES=(3e-6 1e-5)
CONDITION=uniform
SIZE=5000
SEED=0
OUT_ROOT=runs_tune_5000

# Same for every task.
STEPS=940             # about 3 epochs of 5000 examples at batch size 16
EVAL_EVERY=20         # 47 evaluations; at size 1000 the best point came early
WARMUP_STEPS=94       # 10% of the steps

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
