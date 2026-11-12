# machine_translation

Chinese→English translation for CS4248 (NUS). We fine-tune M2M-100 (418M) on ALMA data and compare
four ways of sampling the training data (`uniform`, `length`, `joint`, `loss`) at training-set
sizes 1000, 2000, 5000 and full, with seeds 0, 1 and 2.

Experiment details and learning-rate tuning: [explanations/EXPERIMENTS.md](explanations/EXPERIMENTS.md).

## Folder structure

| Path | Contents |
|---|---|
| `scripts/` | Python code: data preparation, training (`train.py`), evaluation |
| `jobs/` | Slurm job files and launchers for the cluster |
| `data/raw/` | Original input files (Canvas test sets, Tatoeba). Everything else in `data/` is generated |
| `runs/` | Final training runs: `run_info.json` and `dev_curve.csv` per run (checkpoints not committed) |
| `runs_tune*/` | Learning-rate tuning runs |
| `runs_alma_news/` | Runs trained on ALMA + News Commentary |
| `runs_decode_ablation/` | Greedy vs beam search on dev |
| `results/alma/` | Test-set results per run: `<run>/beam{1,4}/{wmt22,tatoeba}/` (predictions, BLEU/chrF, COMET) |
| `results/_evaluation_workflow/` | Evaluation plan, frozen code and `summary.csv` of all test results |
| `notebooks/` | Analysis: `results.ipynb` (main results), `tuning`, `decoding`, `difficulty_binning`, `loss_difficulty` |
| `analysis/` | Error analysis |
| `figures/` | Plots |
| `tests/` | Unit tests |
| `explanations/` | Extra documentation |

## How to run

Run everything from the repository root, on the cluster.

**1. Setup**

```bash
conda env create -f environment.yml
conda activate mt
bash scripts/prepare_data.sh         
sbatch jobs/compute_loss.sbatch       # loss scores for the `loss` condition (needs GPU)
```

**2. Train** all conditions × sizes × seeds (skips runs that are already done):

```bash
CONDITIONS="uniform length joint loss" bash jobs/run_experiments.sh
```

Single run: `sbatch jobs/train1.sbatch --condition joint --size 1000 --seed 0 --lr 3e-6 --steps 300 --warmup_steps 30 --eval_every 10`
(settings per size are in [EXPERIMENTS.md](explanations/EXPERIMENTS.md#final-settings-every-condition-and-seed)).

**3. Evaluate** on the test sets. All results are already committed in `results/alma/`, so this is
only needed to recompute them.

You need the trained checkpoints: a folder containing `<condition>_n<size>_seed<seed>/best/` and
`run_info.json` for all 48 runs (what step 2 writes to `runs/`). Pass it as `--runs_dir`; if the
runs are split over several folders, list them all. Also set `COMET_PYTHON` and `HF_HOME`
([COMET setup](explanations/EVALUATION.md#comet-environment)).

```bash
WORK=results/eval_new      # new folder for this evaluation (an existing plan is never overwritten)
python scripts/run_remaining_evaluations.py plan --runs_dir /path/to/runs --work_dir $WORK
sbatch --array=$(cat $WORK/pending.txt) jobs/remaining_evaluations.sbatch --work_dir $WORK
python scripts/run_remaining_evaluations.py audit --work_dir $WORK    # when the jobs finish -> $WORK/summary.csv
```

Results already in `results/alma/` are skipped, so `pending.txt` only lists what is missing.

**4. Analyse:** open `notebooks/results.ipynb`.

Optional:
- Baseline (no fine-tuning): `sbatch jobs/eval_baseline.sbatch`
- Greedy vs beam on dev: `RUNS_DIR=/path/to/runs sbatch jobs/compare-decoding.sbatch` ([details](explanations/DECODING_COMPARISON.md))
- Evaluate a single checkpoint: [explanations/EVALUATION.md](explanations/EVALUATION.md#evaluating-a-single-checkpoint)
- Error analysis: [analysis/QUANTITATIVE_WORKFLOW.md](analysis/QUANTITATIVE_WORKFLOW.md)
- Tests: `python -m unittest discover -s tests`
