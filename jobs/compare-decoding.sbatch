#!/bin/bash
#SBATCH --job-name=mt-decode
#SBATCH --partition=gpu
#SBATCH --gpus=h100-47:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-8%2
#SBATCH --output=logs/slurm/decode-%A-%a.out

# Greedy vs beam 4 on ALMA dev for the 9 full-data runs (uniform/length/joint x 3 seeds).
# Run from the repository root; RUNS_DIR is the folder with <condition>_nfull_seed<seed>/best/:
#   RUNS_DIR=/path/to/runs sbatch jobs/compare-decoding.sbatch

set -eo pipefail
: "${RUNS_DIR:?set RUNS_DIR to the folder with the trained runs}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mt
cd "$SLURM_SUBMIT_DIR"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

conditions=(uniform length joint)
condition="${conditions[$((SLURM_ARRAY_TASK_ID / 3))]}"
seed=$((SLURM_ARRAY_TASK_ID % 3))

python -u scripts/compare_decoding.py \
  --run "${RUNS_DIR}/${condition}_nfull_seed${seed}" \
  --out_dir "runs_decode_ablation/${SLURM_ARRAY_JOB_ID}"
