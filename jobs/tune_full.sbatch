#!/bin/bash
# Learning-rate tuning at the full training size (15,362 examples), uniform sampling only, seed 0.
#
# This is a Slurm *job array*: one submission starts one task per learning rate:
#
#     task 0: lr = 3e-6
#     task 1: lr = 1e-5
#
# Round 2: longer budget. Round 1 (lr 3e-6 / 1e-5 / 3e-5, 2900 steps, in runs_tune_full/)
# picked lr 1e-5 (dev loss 2.106, BLEU 16.3), but its best checkpoint was the last step.
# That can come from the learning rate decaying to 0 rather than from too few steps, so
# this round trains for ~5 epochs to check. 3e-6 is repeated because a longer budget can
# favour a smaller learning rate. 3e-5 is dropped: it was clearly worse.
#
# Decision rule, fixed before the results: use 4800 steps for full only if the lr 1e-5
# run beats 2.106 dev loss by about 0.01 or more. Otherwise keep 2900 steps.
#
# Why uniform only: at size 1000 (runs_tune2/) all three conditions chose the same
# learning rate, so tuning one condition is enough. The chosen learning rate is then
# used for every condition at this size.
#
# Round 1 also showed that the best learning rate depends on size (3e-6 at 1000,
# 1e-5 at full), so size 5000 gets tuned too.
#
# Budget: 4800 steps x batch 16 = about 5 epochs of the full data (round 1: 2900 steps,
# about 3 epochs). Warm-up is 10% of the steps.
#
# The run with the chosen learning rate and budget also counts as the final uniform
# seed-0 run, so checkpoints are kept: the winner's is needed later for test evaluation.
# Delete the others once the choice is made (each is ~2 GB).
#
# Run from the repository root:
#     sbatch jobs/tune_full.sbatch
# Re-run only some tasks (e.g. after a failure):
#     sbatch --array=1 jobs/tune_full.sbatch
# Both tasks run at once: the cluster allows 2 GPUs per user (QOSMaxGRESPerUser).
#
# Outputs (one folder per learning rate, so runs never overwrite each other):
#     runs_tune_full2/lr<LR>/uniform_nfull_seed0/{best/, dev_curve.csv, run_info.json}
# Logs:
#     logs/slurm/mt-tune-full-<array job id>_<task id>.out
#
# Based on job_template.sbatch from the Unofficial SoC Cluster Guide.

#SBATCH --job-name=mt-tune-full
#SBATCH --array=0-1           # 2 tasks, so both run at once on our 2 GPUs
#SBATCH --partition=gpu-long
#SBATCH --time=08:00:00        # generous: round 1 fit in 4 h, this is 1.7x the steps
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000

set -euo pipefail

# --- What to tune ----------------------------------------------------------------
LEARNING_RATES=(3e-6 1e-5)
CONDITION=uniform
SIZE=full
SEED=0
OUT_ROOT=runs_tune_full2   # round 1 is in runs_tune_full/; a new folder keeps it untouched

# Same for every task.
STEPS=4800            # about 5 epochs of 15,362 examples at batch size 16
EVAL_EVERY=100        # 48 evaluations along the run
WARMUP_STEPS=480      # 10% of the steps

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
