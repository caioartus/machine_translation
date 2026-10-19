#!/bin/bash
# Run the whole data preparation, in order (a few minutes on CPU, no GPU needed).
#
# Run from anywhere, with the `mt` conda environment active:
#     conda activate mt
#     bash scripts/prepare_data.sh
#
# Needs internet access: it downloads ALMA, the M2M-100 tokenizer and the WMT22
# metadata. On the cluster, run it on the login node, not inside a Slurm job
# (compute nodes may not have internet access).
#
# Safe to re-run: every step overwrites its own outputs.
# Stops at the first step that fails, so a later step never runs on missing data.

set -euo pipefail

# Work from the repository root (the folder above this script), wherever we are called from.
cd "$(dirname "$0")/.."

run_step() {
    echo
    echo "=== $1 ==="
    python -u "scripts/$1"
}

run_step audit_alma.py           # 1. measure raw ALMA, flag problems   -> data/audit/
run_step prepare_alma.py         # 2. clean ALMA, split train / dev     -> data/processed/alma_*
run_step prepare_test_sets.py    # 3. build the test sets               -> data/processed/*_test.jsonl
run_step compute_difficulty.py   # 4. length and rarity per example     -> data/features/*.csv
run_step make_bins.py            # 5. difficulty groups and reports     -> data/features/bins*
run_step make_subsets.py         # 6. training subsets per seed / size  -> data/subsets/

echo
echo "Data preparation finished."
