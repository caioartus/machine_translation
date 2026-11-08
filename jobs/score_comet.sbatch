#!/bin/bash
#SBATCH --job-name=mt-comet
#SBATCH --partition=gpu-long
#SBATCH --gpus=h100-47:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs/slurm
exec > "logs/slurm/mt-comet-${SLURM_JOB_ID:-$$}.out" 2>&1

# Set these before submitting the job.
: "${COMET_PYTHON:?Set COMET_PYTHON to the working COMET Python executable}"
: "${HF_HOME:?Set HF_HOME to the prepared Hugging Face cache}"
test -x "$COMET_PYTHON"

# Use compute-node temporary storage to avoid NFS cleanup errors.
export TMPDIR
TMPDIR=$(mktemp -d "/tmp/mt-comet-${SLURM_JOB_ID:-$$}-XXXXXX")

cleanup() {
    rmdir "$TMPDIR" 2>/dev/null ||
        echo "Temporary directory retained because it is not empty: $TMPDIR"
}
trap cleanup EXIT

export HF_HOME
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "Started $(date) on $(hostname)"
echo "Python: $COMET_PYTHON"
echo "Model cache: $HF_HOME"
echo "Temporary directory: $TMPDIR"

"$COMET_PYTHON" -u scripts/score_comet.py "$@"

echo "Finished $(date)"
