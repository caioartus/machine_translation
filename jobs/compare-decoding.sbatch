#!/bin/bash
#SBATCH --job-name=mt-decode
#SBATCH --partition=gpu
#SBATCH --gpus=h100-47:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-8%2
#SBATCH --output=logs/slurm/decode-%A-%a.out

set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate CS4248
cd "$HOME/machine_translation"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

conditions=(uniform length joint)
condition="${conditions[$((SLURM_ARRAY_TASK_ID / 3))]}"
seed=$((SLURM_ARRAY_TASK_ID % 3))

python -u scripts/compare_decoding.py \
  --run "runs_my_full/${condition}_nfull_seed${seed}" \
  --out_dir "runs_decode_ablation/${SLURM_ARRAY_JOB_ID}"
