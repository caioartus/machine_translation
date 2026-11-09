#!/bin/bash
# Evaluate every final checkpoint on the test sets, greedy and beam search, then score COMET.
# One Slurm job per checkpoint; inside it, every test set x beam setting runs in sequence.
# Run from anywhere; it changes to the repository root itself.
#
# Step 1, translations + BLEU/chrF (jobs/evaluate_run.sbatch -> scripts/evaluate.py):
#   DRY_RUN=1 bash jobs/run_evaluation.sh                 # preview
#   bash jobs/run_evaluation.sh                           # submit
#   RUNS_DIR=/path/to/runs bash jobs/run_evaluation.sh    # checkpoints stored elsewhere
#   CONDITIONS=loss SIZES="1000 2000" bash jobs/run_evaluation.sh   # a subset
#
# Step 2, COMET on every finished translation (jobs/comet_run.sbatch -> scripts/score_comet.py):
#   export COMET_PYTHON=... HF_HOME=...                   # see EVALUATION.md
#   MODE=comet bash jobs/run_evaluation.sh
#
# Output, per checkpoint <name> (e.g. loss_n1000_seed0, or "pretrained" for the baseline):
#   $OUT_ROOT/<name>/<test_set>_beam<k>/{predictions,metrics,metadata}.json(l)   + comet.json
#   + nbest.jsonl.gz: every beam candidate (all k of them) with its model log-probability
#
# Safe to re-run: finished outputs (metadata status "complete" / comet.json present) are skipped,
# and a checkpoint whose job is still queued or running is not submitted again. An unfinished
# output folder from a failed job is renamed to <folder>.incomplete-<time> and redone.
#
# Settings (environment variables):
#   RUNS_DIR    where <name>/best/ checkpoints are            (default: runs)
#   OUT_ROOT    where results go                             (default: outputs/eval/final)
#   CONDITIONS / SIZES / SEEDS                               (default: all 4 / all 4 / 0 1 2)
#   TEST_SETS   names of data/processed/<name>_test.jsonl    (default: wmt22 tatoeba)
#   BEAMS       num_beams values; 1 = greedy                 (default: 1 4)
#   BASELINE    1 = also evaluate the pretrained model       (default: 1)
#   SAVE_NBEST  1 = also write nbest.jsonl.gz (extra decoding pass) (default: 1)
#   TIME        time limit per job                           (default: 01:00:00 translate, 00:30:00 comet)

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${MODE:-translate}"
RUNS_DIR="${RUNS_DIR:-runs}"
OUT_ROOT="${OUT_ROOT:-outputs/eval/final}"
CONDITIONS="${CONDITIONS:-uniform length joint loss}"
SIZES="${SIZES:-1000 2000 5000 full}"
SEEDS="${SEEDS:-0 1 2}"
TEST_SETS="${TEST_SETS:-wmt22 tatoeba}"
BEAMS="${BEAMS:-1 4}"
BASELINE="${BASELINE:-1}"
SAVE_NBEST="${SAVE_NBEST:-1}"
EVAL_FLAGS=""; [ "$SAVE_NBEST" = "1" ] && EVAL_FLAGS="--save_nbest"
DRY_RUN="${DRY_RUN:-0}"

case "$MODE" in
    translate) TIME="${TIME:-01:00:00}" ;;
    comet)
        TIME="${TIME:-00:30:00}"
        : "${COMET_PYTHON:?Set COMET_PYTHON (see EVALUATION.md)}"
        : "${HF_HOME:?Set HF_HOME (see EVALUATION.md)}"
        ;;
    *) echo "MODE must be translate or comet" >&2; exit 1 ;;
esac

for set_name in $TEST_SETS; do
    [ -f "data/processed/${set_name}_test.jsonl" ] || { echo "missing data/processed/${set_name}_test.jsonl" >&2; exit 1; }
done

# name<TAB>model for every checkpoint to evaluate.
checkpoints() {
    [ "$BASELINE" = "1" ] && printf 'pretrained\tfacebook/m2m100_418M\n'
    for size in $SIZES; do for seed in $SEEDS; do for cond in $CONDITIONS; do
        name="${cond}_n${size}_seed${seed}"
        printf '%s\t%s\n' "$name" "$RUNS_DIR/$name/best"
    done; done; done
}

is_complete() { grep -q '"status": "complete"' "$1/metadata.json" 2>/dev/null; }

in_queue() {  # job with this name queued or running for this user
    command -v squeue >/dev/null && [ -n "$(squeue -h -u "$USER" -n "$1" -o %i 2>/dev/null)" ]
}

mkdir -p logs/slurm
n_sub=0; n_skip=0; n_missing=0
while IFS=$'\t' read -r name model; do
    if [ "$MODE" = "translate" ]; then
        job_name="mt-eval-$name"
        if [ "$name" != "pretrained" ] && [ ! -d "$model" ]; then
            echo "missing $name (no checkpoint at $model)"; n_missing=$((n_missing + 1)); continue
        fi
        todo=0
        for set_name in $TEST_SETS; do for k in $BEAMS; do
            is_complete "$OUT_ROOT/$name/${set_name}_beam$k" || todo=$((todo + 1))
        done; done
        cmd=(sbatch --parsable --job-name="$job_name" --time="$TIME" jobs/evaluate_run.sbatch
             "$model" "$OUT_ROOT/$name" "$TEST_SETS" "$BEAMS" "$EVAL_FLAGS")
    else
        job_name="mt-comet-$name"
        todo=0
        for set_name in $TEST_SETS; do for k in $BEAMS; do
            dir="$OUT_ROOT/$name/${set_name}_beam$k"
            is_complete "$dir" && [ ! -f "$dir/comet.json" ] && todo=$((todo + 1))
        done; done
        cmd=(sbatch --parsable --job-name="$job_name" --time="$TIME" jobs/comet_run.sbatch
             "$OUT_ROOT/$name" "$TEST_SETS" "$BEAMS")
    fi

    if [ "$todo" -eq 0 ]; then
        echo "skip    $name (nothing to do)"; n_skip=$((n_skip + 1)); continue
    fi
    if in_queue "$job_name"; then
        echo "skip    $name (job already queued or running)"; n_skip=$((n_skip + 1)); continue
    fi
    if [ "$DRY_RUN" = "1" ]; then
        echo "dry     $name ($todo to do): ${cmd[*]}"
    else
        job_id="$("${cmd[@]}")"
        echo "submit  $name ($todo to do)  job $job_id"
        printf '%s\t%s\t%s\t%s\n' "$job_id" "$MODE" "$name" "$(date -Is)" >> logs/evaluation_submitted.tsv
    fi
    n_sub=$((n_sub + 1))
done < <(checkpoints)

echo "done ($MODE): $n_sub submitted (or would be), $n_skip skipped, $n_missing checkpoints missing"
