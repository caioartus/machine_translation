#!/bin/bash
# Submit the final experiments: 3 conditions x 4 sizes x 3 seeds (36 runs, see README).
# Run from anywhere; it changes to the repository root itself.
#
#   bash jobs/run_experiments.sh                    # submit everything that is not done yet
#   DRY_RUN=1 bash jobs/run_experiments.sh          # only print what would be submitted
#   SIZES="1000 2000" bash jobs/run_experiments.sh  # a subset (also CONDITIONS=..., SEEDS=...)
#
# Safe to re-run: a run whose <out_dir>/<name>/run_info.json exists is skipped.
# Each run is its own Slurm job (jobs/train1.sbatch), so runs are independent and a failure
# only affects that run. Settings per size are the "Final settings" table in the README.

set -euo pipefail
cd "$(dirname "$0")/.."

SBATCH_FILE="jobs/train1.sbatch"
CONDITIONS="${CONDITIONS:-uniform length joint}"
SIZES="${SIZES:-1000 2000 5000 full}"
SEEDS="${SEEDS:-0 1 2}"
OUT_DIR="${OUT_DIR:-runs}"
DRY_RUN="${DRY_RUN:-0}"

# size -> "lr steps warmup eval_every time_limit"  (identical for every condition and seed)
# Time limits are generous guesses, adjust them to the durations in your tuning logs.
settings() {
    case "$1" in
        1000) echo "3e-6 300 30 10 00:45:00" ;;
        2000) echo "3e-6 375 38 20 01:00:00" ;;
        5000) echo "3e-6 940 94 20 01:30:00" ;;
        full) echo "1e-5 2900 290 100 04:00:00" ;;
        *) echo "unknown size: $1" >&2; exit 1 ;;
    esac
}

mkdir -p logs
n_submitted=0; n_skipped=0
for size in $SIZES; do
    read -r lr steps warmup eval_every time_limit <<< "$(settings "$size")"
    for seed in $SEEDS; do
        for cond in $CONDITIONS; do
            name="${cond}_n${size}_seed${seed}"
            target="$OUT_DIR/$name"

            if [ -f "$target/run_info.json" ]; then
                echo "skip   $name (already done)"; n_skipped=$((n_skipped + 1)); continue
            fi

            cmd=(sbatch --parsable --job-name="mt-$cond-n$size-s$seed" --time="$time_limit"
                 "$SBATCH_FILE" --condition "$cond" --size "$size" --seed "$seed"
                 --lr "$lr" --steps "$steps" --warmup_steps "$warmup" --eval_every "$eval_every"
                 --out_dir "$OUT_DIR")
            if [ "$DRY_RUN" = "1" ]; then
                echo "dry    ${cmd[*]}"
            else
                job_id="$("${cmd[@]}")"
                echo "submit $name  job $job_id  ($time_limit)"
                printf '%s\t%s\t%s\n' "$job_id" "$name" "$(date -Is)" >> logs/experiments_submitted.tsv
            fi
            n_submitted=$((n_submitted + 1))
        done
    done
done
echo "done: $n_submitted submitted (or would be), $n_skipped skipped"
