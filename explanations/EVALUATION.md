# Evaluation

How to run it is in the README. This page covers what it produces and how to set up COMET.

## What gets evaluated

Every final run (4 conditions × 4 sizes × 3 seeds = 48) on both official test sets
(`wmt22_test.jsonl`, `tatoeba_test.jsonl`) with greedy (`beam1`) and beam 4 (`beam4`) decoding:
192 combinations. WMT22 is scored against Annotator A (`en`).

`scripts/run_remaining_evaluations.py`:
- `plan --runs_dir <dir> [<dir> ...]`: finds every run in the given folder(s) (the first folder
  holding `<run>/run_info.json` wins), checks its settings match the final settings, freezes a copy
  of the evaluation code in `<work_dir>/code/`, and writes the missing task ids to
  `<work_dir>/pending.txt`. A plan is never overwritten: use a new `--work_dir` to plan again.
- `task <i>` (run by `jobs/remaining_evaluations.sbatch`): translates and scores task `i`, then
  validates the output.
- `audit`: re-validates everything and writes `<work_dir>/summary.csv`.

Finished outputs are validated and skipped, so planning again only queues what is missing.

## Outputs

`results/alma/<run>/beam<k>/<test_set>/`:

| File | Content |
|---|---|
| `predictions.jsonl` | Source, reference and model translation per sentence |
| `metrics.json` | BLEU and chrF (sacreBLEU) |
| `comet.json` | COMET (`Unbabel/wmt22-comet-da`), system and per-sentence scores (raw, not × 100) |
| `metadata.json` | Model, data hash, decoding settings, package versions |

## Evaluating a single checkpoint

```bash
sbatch jobs/evaluate.sbatch --model <run>/best --data data/processed/wmt22_test.jsonl \
  --num_beams 4 --out_dir outputs/eval/<name>
sbatch jobs/score_comet.sbatch --predictions outputs/eval/<name>/predictions.jsonl \
  --output outputs/eval/<name>/comet.json
```

Use `--model facebook/m2m100_418M` for the pretrained baseline. Output folders are never overwritten.

## COMET environment

COMET runs in its own environment so it does not change the training dependencies. The jobs read
two variables:

```bash
export COMET_PYTHON=/path/to/comet-venv/bin/python   # python with unbabel-comet==2.2.7 installed
export HF_HOME=/path/to/huggingface-cache            # cache holding the COMET model
```

`jobs/remaining_evaluations.sbatch` defaults to Jake's shared setup
(`/mnt/scratch/t/teng/comet-work/...`) when these are not set.

New setup: create a Python 3.11 venv, `pip install unbabel-comet==2.2.7`, then download the model
on the login node (compute nodes run offline):

```bash
HF_HUB_DISABLE_XET=1 "$COMET_PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download('Unbabel/wmt22-comet-da')
snapshot_download('xlm-roberta-large', allow_patterns=['*.json', 'sentencepiece.bpe.model'])
"
```

Importing PyTorch can fail on the login node (memory limit); that does not mean the install is
broken. Test it inside a GPU job.
