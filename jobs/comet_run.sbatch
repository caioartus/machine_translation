#!/bin/bash
# COMET for every finished translation of one checkpoint. Submitted by jobs/run_evaluation.sh:
#   sbatch jobs/comet_run.sbatch <out_dir> "<test sets>" "<beams>"
# Scores <out_dir>/<test_set>_beam<k>/predictions.jsonl -> comet.json (skipped if present).
# Same environment as jobs/score_comet.sbatch: COMET_PYTHON and HF_HOME must be exported.

#SBATCH --job-name=mt-comet
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
OUT_DIR="$1"; TEST_SETS="$2"; BEAMS="$3"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs/slurm
exec > "logs/slurm/${SLURM_JOB_NAME:-mt-comet}-${SLURM_JOB_ID:-$$}.out" 2>&1

: "${COMET_PYTHON:?Set COMET_PYTHON to the working COMET Python executable}"
: "${HF_HOME:?Set HF_HOME to the prepared Hugging Face cache}"
test -x "$COMET_PYTHON"

# Use compute-node temporary storage to avoid NFS cleanup errors.
export TMPDIR
TMPDIR=$(mktemp -d "/tmp/mt-comet-${SLURM_JOB_ID:-$$}-XXXXXX")
cleanup() {
    rm -rf "$TMPDIR" 2>/dev/null || echo "Temporary directory retained: $TMPDIR"
}
trap cleanup EXIT

export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false

echo "Started $(date) on $(hostname); out: $OUT_DIR"
failed=0
for set_name in $TEST_SETS; do
    for k in $BEAMS; do
        dir="$OUT_DIR/${set_name}_beam$k"
        if ! grep -q '"status": "complete"' "$dir/metadata.json" 2>/dev/null; then
            echo "== skip $dir (translation not complete)"; continue
        fi
        [ -f "$dir/comet.json" ] && { echo "== skip $dir (comet.json exists)"; continue; }
        echo "== $dir  ($(date +%T))"
        "$COMET_PYTHON" -u scripts/score_comet.py --predictions "$dir/predictions.jsonl" \
            --output "$dir/comet.json" --batch_size 8 --gpus 1 \
            || { echo "!! FAILED: $dir"; failed=1; }
    done
done
echo "Finished $(date)"
exit "$failed"
