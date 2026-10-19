# machine_translation
In this project we aim to build a fast and efficient machine translation system to translate from chinese to english. This project was done in the scope of the Introduction to Large Language Models class (CS4248) at NUS. 

## Training

Training runs on the cluster through Slurm. Run everything from the repository root.
Arguments after the `.sbatch` file are forwarded to `scripts/train.py`.

First, prepare the data. This runs every data preparation step in order. Run it on the login
node, since it downloads ALMA and the tokenizer and compute nodes may have no internet access.

```bash
git pull
conda env update -n mt -f environment.yml --prune   # adds pandas / matplotlib / sacrebleu, pinned
conda activate mt
bash scripts/prepare_data.sh
```

### Submit a run

```bash
# one run: sampling condition, training-set size, seed
sbatch jobs/train1.sbatch --condition joint --size 1000 --seed 0

# longer time limit (default is 30 minutes), set on the command line
sbatch --time=02:00:00 jobs/train1.sbatch --condition uniform --size 5000 --seed 0

# change hyperparameters; use a separate --out_dir per setting so runs don't overwrite each other
sbatch jobs/train1.sbatch --condition uniform --size 1000 --seed 0 --lr 3e-6 --eval_every 25 --out_dir runs_lr/lr3e-6

# no arguments: quick smoke test (20 steps, output in runs_test/)
sbatch jobs/train1.sbatch
```

### Arguments

| Argument | Default | Meaning |
|---|---|---|
| `--condition` | required | Sampling condition: `uniform`, `length` or `joint` |
| `--size` | `1000` | Training subset: `1000`, `2000`, `5000` or `full` |
| `--seed` | `0` | Picks the subset (`seed{S}_n{N}.txt`) and seeds sampler, dropout and init |
| `--steps` | `1000` | Optimizer steps. Must be identical across conditions for a given size |
| `--batch_size` | `16` | Examples per step. Must be identical across conditions |
| `--lr` | `3e-5` | Peak learning rate (linear warm-up, then linear decay to 0) |
| `--warmup_steps` | `50` | Steps of linear warm-up |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--eval_every` | `100` | Compute dev loss (and consider saving a checkpoint) every k steps |
| `--cap` | `3.0` | Maximum sampling weight relative to uniform (see "The conditions" above) |
| `--out_dir` | `runs` | Parent folder for results |

The defaults for `steps`, `batch_size`, `lr` and `cap` are working placeholders, not final
decisions. Every condition must use the same values at a given size.

### Monitor a job

```bash
squeue --me                                  # your queued and running jobs (and their job ids)
tail -f logs/slurm/mt-train-<job_id>.out     # follow the log live (Ctrl-C stops following, not the job)
scancel <job_id>                             # cancel a job
```

The log shows the dev loss at step 0 (the pretrained model), then train and dev loss at every
`--eval_every` steps, and finally the best step and greedy dev BLEU.

### Results

Each run writes to `<out_dir>/<condition>_n<size>_seed<seed>/`:

| File | Content |
|---|---|
| `best/` | Checkpoint with the lowest dev loss (model and tokenizer) |
| `dev_curve.csv` | `step`, `train_loss` (mean since the last eval), `dev_loss`. Step 0 is the pretrained model |
| `run_info.json` | All settings, `best_step`, `best_dev_loss`, `dev_bleu_greedy` |

Checkpoints are about 1.9 GB each, so check your disk quota before running many jobs.

### Baseline (no fine-tuning)

To compare a run against the unchanged pretrained model on `alma_dev`:

```bash
sbatch jobs/eval_baseline.sbatch                                          # -> baselines/pretrained.json
sbatch jobs/eval_baseline.sbatch --model runs/joint_n1000_seed0/best      # score a trained checkpoint
```

It reports `dev_loss` and `dev_bleu_greedy`, computed the same way as in `run_info.json`.

### Hyperparameter tuning

We tune only the learning rate (and, at full size, the step budget), on the dev set, seed 0.
Batch size 16, weight decay, dropout and cap stay fixed; warm-up is 10% of the steps.
At size 1000 all three conditions are tuned with the same grid; they agreed, so larger
sizes are tuned with uniform only and the result is used for every condition.
Look at any round's results in `notebooks/tuning.ipynb` (set `TUNE_DIR` to its folder).

**Log of every tuning run** (each job file is edited in place between rounds; git history keeps the old versions):

| Round | Job | Size | Conditions | Learning rates | Steps | Warm-up | Eval every | Results | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| 1000-1 | `jobs/tune.sbatch` | 1000 | all 3 | 1e-5, 3e-5, 1e-4 | 1000 | 50 | 100 | `runs_tune/` | 3e-5 and 1e-4 overfit at once; 1e-5 best for all conditions, at step 100 |
| 1000-2 | `jobs/tune.sbatch` | 1000 | all 3 | 3e-6, 5e-6, 1e-5 | 300 | 30 | 10 | `runs_tune2/` | 3e-6 lowest dev loss for all conditions; best step 130-160 |
| full-1 | `jobs/tune_full.sbatch` | full | uniform | 3e-6, 1e-5, 3e-5 | 2900 (~3 epochs) | 290 | 100 | `runs_tune_full/` | 1e-5 best; best step = last step (2900) |
| full-2 | `jobs/tune_full.sbatch` | full | uniform | 3e-6, 1e-5 | 4800 (~5 epochs) | 480 | 100 | `runs_tune_full2/` | no real gain from more steps (1e-5: 2.110, worse; 3e-6: 2.099, below the 0.01 threshold); keep 2900 |
| 5000-1 | `jobs/tune_5000.sbatch` | 5000 | uniform | 3e-6, 1e-5 | 940 (~3 epochs) | 94 | 20 | `runs_tune_5000/` | running |

Round 1000-1 showed the best point is within the first 100 steps and that the best learning rate was
the lowest one tried, so round 1000-2 tries lower rates and evaluates more often.

Round 2, best dev loss / dev BLEU (greedy) per run:

| lr | uniform | length | joint |
|---|---|---|---|
| 3e-6 | **2.174** / 14.98 | **2.181** / 14.87 | **2.186** / 14.95 |
| 5e-6 | 2.176 / 15.08 | 2.184 / 14.74 | 2.187 / 14.80 |
| 1e-5 | 2.188 / 14.90 | 2.189 / 14.64 | 2.198 / 14.72 |

Pretrained model (step 0): dev loss 2.315, BLEU 14.1.

**Decision for size 1000: lr 3e-6, 300 steps.** It has the lowest dev loss for every condition,
and its dev loss stays flat up to step 300 instead of rising. BLEU differences between
learning rates are at most 0.3, within seed noise, so at this size the learning rate matters
little once it is 1e-5 or below.

Round full-1, uniform, best dev loss / dev BLEU (greedy) / best step:

| lr | uniform |
|---|---|
| 3e-6 | 2.110 / 15.71 / 2600 |
| 1e-5 | **2.106** / **16.33** / 2900 |
| 3e-5 | 2.161 / 15.90 / 2900 |

1e-5 is best on both dev loss and BLEU, and sits in the middle of the grid. 3e-5 is too high:
dev loss jumps after step 100 and only slowly recovers. So the best learning rate depends on
size (3e-6 at 1000, 1e-5 at full), and size 5000 will be tuned too.

The best 1e-5 checkpoint is the last step, but the curve is nearly flat by then (2.116 at step
2000, 2.106 at 2900) and the learning rate decays to 0, so this does not prove the budget is too
short. Round full-2 checks it with ~5 epochs. **Decision rule, fixed before the results:** use
4800 steps at full only if lr 1e-5 beats 2.106 dev loss by about 0.01 or more; otherwise keep 2900.

Round full-2, uniform, 4800 steps, best dev loss / dev BLEU (greedy) / best step:

| lr | uniform |
|---|---|
| 3e-6 | 2.099 / 16.06 / 4600 |
| 1e-5 | 2.110 / 16.06 / 3300 |

With more steps, lr 1e-5 got slightly worse (2.110 vs 2.106), so the rule says keep 2900. 3e-6 gets
the lowest dev loss, but only by 0.007, below the threshold, and with 0.27 lower BLEU. Dev loss is
flat after about step 3000 in both runs, so the extra steps buy nothing.

**Decision for full size: lr 1e-5, 2900 steps.** The full-1 run with these settings
(`runs_tune_full/lr1e-5/uniform_nfull_seed0`) is also the final uniform seed-0 run.



## 🚧 Temporary: handoff from data preparation to model training

> Temporary section (2026-10-13). Remove once training code exists and this is folded into the docs.
> Everything below is ready to use, except the items under "Not ready yet".

### 1. Build the data (once per machine, a few minutes on CPU)

```bash
git pull
conda env update -n mt -f environment.yml --prune   # adds pandas / matplotlib / sacrebleu, pinned
conda activate mt
bash scripts/prepare_data.sh
```

### 2. What you need, and nothing else

| What | File(s) | Use it for |
|---|---|---|
| Training pairs | `data/processed/alma_train.jsonl` | Text (`zh`, `en`), joined on `id` |
| Training subsets | `data/subsets/seed{S}_n{N}.txt`, N ∈ {1000, 2000, 5000, full}, S ∈ {0, 1, 2} | Which ids to train on. **Run seed S on the subsets of seed S.** |
| Difficulty features | `data/features/alma_train.csv` | Pass through `assign_bins()` to get `length_bin` and `cell` |
| Dev set | `data/processed/alma_dev.jsonl` (1,002 pairs) | Checkpoint selection and any tuning |
| Test sets | `data/processed/{wmt22,tatoeba}_test.jsonl` | **Final evaluation only. Never for selection or tuning.** |

Every `.jsonl` line looks like `{"id": "alma-train-000042", "source": "alma", "zh": "...", "en": "..."}`.

### 3. Load one training subset with its groups

```python
import pandas as pd
from data_utils import assign_bins, read_jsonl      # works for scripts in scripts/, run from the repo root

seed, size = 0, 1000                                 # size: 1000 | 2000 | 5000 | "full"
ids = open(f"data/subsets/seed{seed}_n{size}.txt").read().split()
text = pd.DataFrame(read_jsonl("data/processed/alma_train.jsonl"))
feats = assign_bins(pd.read_csv("data/features/alma_train.csv"))
train = text.merge(feats, on="id").set_index("id").loc[ids].reset_index()
# columns: id, zh, en, src_len, rarity_mean, ..., length_bin, rarity_bin, cell
dev = read_jsonl("data/processed/alma_dev.jsonl")
```

### 4. The conditions are only a change of sampling weights

The model, loss, optimizer and number of steps stay identical across conditions. Only the
weights given to `WeightedRandomSampler` change:

```python
import torch
from torch.utils.data import WeightedRandomSampler

def balanced_weights(groups, cap=3.0):
    """Per-example weight relative to uniform sampling: every group gets the same total
    probability, but no example is drawn more than `cap` times as often as under uniform."""
    n_groups = groups.nunique()
    group_size = groups.map(groups.value_counts())
    ratio = (len(groups) / n_groups) / group_size
    return ratio.clip(upper=cap)

if condition == "uniform":
    weights = pd.Series(1.0, index=train.index)
elif condition == "length":                          # 4 groups
    weights = balanced_weights(train["length_bin"])
elif condition == "joint":                           # 8 groups (length x rarity)
    weights = balanced_weights(train["cell"])

sampler = WeightedRandomSampler(torch.tensor(weights.values, dtype=torch.double),
                                num_samples=num_steps * batch_size, replacement=True,
                                generator=torch.Generator().manual_seed(seed))
loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate)
```

What this does on `seed0_n1000` (share of training draws, %):

| Group | Natural | Length-balanced | | Cell | Natural | Joint |
|---|---|---|---|---|---|---|
| short | 8.9 | 25.0 | | short/rare | 1.1 | 4.0 (capped at 3×) |
| mid_short | 35.1 | 25.0 | | long/rare | 1.7 | 6.2 (capped at 3×) |
| mid_long | 46.4 | 25.0 | | mid_long/typical | 41.6 | 15.3 |
| long | 9.6 | 25.0 | | … | | |

### 5. Rules that keep the comparison fair

- **Matched budget:** for a given training size, every condition runs the **same number of
  optimizer steps with the same batch size** (that is why the sampler draws `num_steps * batch_size`
  examples with replacement, instead of looping over "epochs").
- **Checkpoint selection on `alma_dev` only**, the same way for every condition (e.g. lowest
  dev loss or best dev BLEU, evaluated every k steps).
- **`max_length=256`** for source and target (`simple_train.py` already does this). The cleaning
  removed every pair longer than that, so nothing is truncated.
- **Seeds:** seed S picks the subset *and* seeds the training run (sampler, dropout, init of anything new).
- Translate with `forced_bos_token_id=tokenizer.get_lang_id("en")` and `tokenizer.src_lang = "zh"`.
- Save, for every run: condition, size, seed, steps, the selected checkpoint, and the dev curve.

### 6. Not ready yet / still to decide

- **Loss-based condition:** needs per-example loss under the pretrained model (step 7, a short
  GPU job, not written yet). Its weights will come from a new column in `alma_train.csv`.
- **Sampling policy details (owner: Xingxia):** `cap=3.0` and "equal share per group" above are a
  working default to agree on, not a final decision.
- **Design note for the joint condition:** balancing over the 8 cells also changes the length mix
  (e.g. short goes from 8.9% to 19.3%, not to the 25% of the length condition). So "joint minus
  length" is not purely the effect of rarity. Options: add a rarity-only condition
  (`balanced_weights(train["rarity_bin"])`), or report it as a limitation.

### 7. For evaluation afterwards (Jake)

- Official numbers: WMT22 against Annotator A (`en`), and Tatoeba; BLEU (sacreBLEU, 13a) and
  COMET `Unbabel/wmt22-comet-da`, as in the course guidelines. chrF is an extra.
- Breakdowns: `assign_bins(pd.read_csv("data/features/wmt22_test.csv"))` gives each test sentence
  the same groups as training. WMT22 also has `domain`.
- `tatoeba_challenge_test.jsonl` (4,758 pairs) is supplementary: use it to check whether
  differences between conditions on short sentences are significant (`sacrebleu --paired-bs`),
  never as a headline number.

## Setup

```bash
conda env create -f environment.yml   # versions are pinned to match the cluster
conda activate mt
```

## Data

Original input files are committed in `data/raw/`; everything else in `data/` is generated by
the scripts in `scripts/` and not committed.

### 1. Raw inputs (already in the repository)

- `data/raw/canvas/`: the files from the CS4248 **Canvas Project folder**, unchanged. The course
  guidelines define the official test sets as these files: `wmttest2022.zh` +
  `wmttest2022.AnnA.en` (WMT22) and `tatoeba.zh` + `tatoeba.en` (Tatoeba).
  Canvas also provides `train.zh-en.{zh,en}`, kept for reference only: we verified it is identical
  to ALMA-Human-Parallel's `train` split (apart from a trailing space on 8 lines), and the scripts
  download ALMA from Hugging Face.
- `data/raw/tatoeba/`: the pinned Tatoeba Challenge test files used for the supplementary set
  (the script re-downloads them only if missing).

### 2. Build everything

Run from the repository root, in this order:

```bash
python scripts/audit_alma.py          # optional: statistics and plots -> data/audit/
python scripts/prepare_alma.py        # clean ALMA -> data/processed/alma_{train,dev}.jsonl
python scripts/prepare_test_sets.py   # test sets  -> data/processed/<name>_test.jsonl
python scripts/compute_difficulty.py  # length / rarity features -> data/features/<split>.csv
python scripts/make_bins.py           # difficulty groups report + heatmap -> data/features/
python scripts/make_subsets.py        # nested training subsets -> data/subsets/seed{S}_n{N}.txt
```

**Difficulty groups.** Bins are defined once in `scripts/data_utils.py` and assigned with
`assign_bins()`, used by both training and evaluation: length (source subword tokens)
short ≤ 15 < mid_short ≤ 30 < mid_long ≤ 60 < long, and rarity (`rarity_mean`) typical ≤ 7.9 < rare,
giving 8 cells such as `short/rare`. Rationale and the distribution of every split over the cells:
`data/features/bins_report.md` and `bins_heatmap.png`.

**Training subsets.** For each seed (0, 1, 2) the training ids are shuffled once and a subset of
size n (1000, 2000, 5000, full) is the first n ids, so subsets are nested. Run seed s on the subsets
of seed s. Cell counts per subset: `data/subsets/subsets_report.md`.

Every processed file uses the same JSON Lines format, one pair per line:
`{"id": "alma-train-000042", "source": "alma", "zh": "...", "en": "..."}`.
Difficulty features are stored separately from the text, one CSV per split, joined on `id`
(see the docstring of `scripts/compute_difficulty.py` for the definition of each column).
Cleaning decisions, counts and checks are documented in `data/processed/alma_cleaning_report.md`
and `data/processed/test_sets_report.md` (both generated by the scripts).

| File (`data/processed/`) | Pairs | Role |
|---|---|---|
| `alma_train.jsonl` | 15,362 | Training (cleaned from 15,406) |
| `alma_dev.jsonl` | 1,002 | Dev: checkpoint selection and tuning |
| `wmt22_test.jsonl` | 1,875 | **Official test set** |
| `tatoeba_test.jsonl` | 469 | **Official test set** |
| `tatoeba_challenge_test.jsonl` | 4,758 | Supplementary test set, never a headline result |

### Sources

| Dataset | Use | Version | Links |
|---|---|---|---|
| ALMA-Human-Parallel, `zh-en` (Xu et al., 2024) | Training (`train`) and dev / checkpoint selection (`validation`) | Hugging Face revision `4cee1a0e3e1daac4b50a759e8d8659ee2e6595bc` (downloaded 2026-10-13) | [dataset](https://huggingface.co/datasets/haoranxu/ALMA-Human-Parallel), [zh-en files](https://huggingface.co/datasets/haoranxu/ALMA-Human-Parallel/tree/main/zh-en), [code](https://github.com/fe1ixxu/ALMA) |
| WMT22 general test set, zh→en (Kocmi et al., 2022) | **Official test set.** Text from Canvas (`wmttest2022.zh`, `wmttest2022.AnnA.en` = Annotator A) | Canvas files; identical to `wmt22-news-systems` release v1.1 (`xml/wmttest2022.zh-en.all.xml`), from which we add Annotator B, `domain` and `docid` via sacreBLEU (`wmt22`, `zh-en`) | [repository](https://github.com/wmt-conference/wmt22-news-systems), [archive](https://github.com/wmt-conference/wmt22-news-systems/archive/refs/tags/v1.1.tar.gz), [task page](https://www.statmt.org/wmt22/translation-task.html) |
| Tatoeba (Tiedemann, 2020; Tatoeba corpus, CC-BY 2.0 FR) | **Official test set.** 469 Simplified Chinese pairs chosen by the course | Canvas files (`tatoeba.zh`, `tatoeba.en`) | [Tatoeba](https://tatoeba.org) |
| Tatoeba Challenge test set, eng-zho (Tiedemann, 2020) | **Supplementary** test set: Simplified Mandarin rows (`cmn_Hans`), minus sources that appear in the official Tatoeba set | Release v2023-09-26, commit `6faf2da` | [repository](https://github.com/Helsinki-NLP/Tatoeba-Challenge), [file](https://raw.githubusercontent.com/Helsinki-NLP/Tatoeba-Challenge/6faf2dab0b7b01a0e08a114dbacbb7deac54988d/data/test/eng-zho/test.txt), [file history](https://github.com/Helsinki-NLP/Tatoeba-Challenge/commits/master/data/test/eng-zho/test.txt) |
| Tatoeba Challenge test set, 2020 version | Only to flag supplementary pairs that existed in 2020 (`in_v2020` column) | Commit `17a1b96` | [file](https://raw.githubusercontent.com/Helsinki-NLP/Tatoeba-Challenge/17a1b96/data/test/eng-zho/test.txt) |
| M2M-100 418M tokenizer (Fan et al., 2021) | Subword tokenisation for length / rarity features | — | [model](https://huggingface.co/facebook/m2m100_418M) |

Notes:
- **Headline results use only the official test sets**, with Annotator A references for WMT22
  (as in the course guidelines). `en_ref_b` and the supplementary set are for extra analysis only.
- Why a supplementary set: 469 sentences are too few to tell apart systems that differ by about
  1–2 BLEU. The Tatoeba Challenge set (4,758 similar short sentences) is used to check whether
  differences between our sampling conditions on short sentences are real.
- Why `in_v2020`: M2M-100's authors removed the Tatoeba Challenge test sentences available in 2020
  from pretraining; pairs added later were not removed and may have been seen in pretraining.
- The raw Tatoeba Challenge downloads are kept untouched in `data/raw/tatoeba/` for provenance.
- ALMA is loaded by name and is not yet pinned to the revision above in the scripts.
