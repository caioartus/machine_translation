#!/bin/bash
# One checkpoint, every test set x beam setting, in sequence. Submitted by jobs/run_evaluation.sh:
#   sbatch jobs/evaluate_run.sbatch <model> <out_dir> "<test sets>" "<beams>" ["<extra evaluate.py flags>"]
# Writes <out_dir>/<test_set>_beam<k>/ with scripts/evaluate.py; finished outputs are skipped.

#SBATCH --job-name=mt-eval
#SBATCH --partition=gpu-long
#SBATCH --time=01:00:00
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8000

set -euo pipefail
MODEL="$1"; OUT_DIR="$2"; TEST_SETS="$3"; BEAMS="$4"; EXTRA="${5:-}"   # e.g. --save_nbest

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-mt}"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p "$REPO_ROOT/logs/slurm"
exec > "$REPO_ROOT/logs/slurm/${SLURM_JOB_NAME:-mt-eval}-${SLURM_JOB_ID:-$$}.out" 2>&1
cd "$REPO_ROOT"
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u

echo "Job ${SLURM_JOB_ID:-local} on $(hostname), started $(date)"
echo "model: $MODEL  out: $OUT_DIR  test sets: $TEST_SETS  beams: $BEAMS  extra: $EXTRA"
nvidia-smi

failed=0
for set_name in $TEST_SETS; do
    for k in $BEAMS; do
        target="$OUT_DIR/${set_name}_beam$k"
        if grep -q '"status": "complete"' "$target/metadata.json" 2>/dev/null; then
            echo "== skip $target (complete)"; continue
        fi
        if [ -e "$target" ]; then  # left over from a failed job; evaluate.py refuses existing dirs
            mv "$target" "$target.incomplete-$(date +%Y%m%d-%H%M%S)"
        fi
        echo "== $target  ($(date +%T))"
        python -u scripts/evaluate.py --device cuda --model "$MODEL" \
            --data "data/processed/${set_name}_test.jsonl" --num_beams "$k" --out_dir "$target" $EXTRA \
            || { echo "!! FAILED: $target"; failed=1; }
    done
done

echo "Finished $(date)"
exit "$failed"
