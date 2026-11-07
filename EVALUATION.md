# Evaluation workflow

This workflow exports Chinese-to-English translations and computes BLEU, chrF,
and COMET. Development-set verification has passed on the SoC cluster.
Final-checkpoint evaluation and comparisons across conditions are still pending.

## Files

- scripts/evaluate.py: translation export, BLEU and chrF.
- scripts/score_comet.py: COMET scoring of saved translations.
- jobs/evaluate.sbatch: GPU job for translation.
- jobs/score_comet.sbatch: GPU job for COMET.
- tests/test_evaluation_io.py: lightweight input/output checks.

Run commands from the repository root. Prepared data, model checkpoints,
Python environments, and downloaded model files must be available separately.
Git does not transfer ignored data, checkpoints, or outputs.

Results belong under outputs/, which is ignored by Git. Use fresh output
directories and filenames: the scripts refuse to overwrite existing results.

## Translation environment

The translation job uses the team's mt Conda environment. By default, the job
initialises Conda from $HOME/miniconda3. CONDA_BASE and CONDA_ENV can override
these locations.

For lightweight checks on the login node:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mt
python -m unittest discover -s tests -p 'test_evaluation_io.py' -v
```

Run model inference through allocated compute jobs, not on the login node.

## Development verification

Use an available checkpoint containing model weights and tokenizer files.
The checkpoint below is a 20-step smoke checkpoint, not a final model.

```bash
sbatch jobs/evaluate.sbatch \
  --model runs_test/joint_n1000_seed0/best \
  --data data/processed/alma_dev.jsonl \
  --limit 20 \
  --out_dir outputs/eval/smoke-dev20
```

Read logs/slurm/mt-evaluate-JOBID.out.

For full development evaluation, remove --limit and choose a new output folder:

```bash
sbatch jobs/evaluate.sbatch \
  --model runs_test/joint_n1000_seed0/best \
  --data data/processed/alma_dev.jsonl \
  --out_dir outputs/eval/smoke-dev-full
```

Default decoding is greedy, with batch size 32 and a 256-token maximum.
The evaluator reuses train.translate and forces the English language tag.

Each evaluation produces:

- predictions.jsonl: ordered id, zh, en, and prediction fields.
- metrics.json: BLEU/chrF scores and metric signatures.
- metadata.json: model path, data hash, settings, package versions, row counts,
  source truncation count, and completion status.

Check that metadata status is complete and that the job completed successfully.
For final scoring, partial_dataset must be false.

Compare full-development BLEU with dev_bleu_greedy in the checkpoint's
run_info.json. A 20-example score is not comparable with the full-development
score. This evaluator does not compute development loss.

## COMET environment

COMET is required for final course evaluation. Its environment is separate
from mt to avoid changing the working training dependencies.

Jake's verified setup uses:

- Python 3.11.18
- unbabel-comet 2.2.7
- PyTorch 2.15.0+cu132
- Transformers 4.57.6

Full package inventories are saved locally under
outputs/eval/verification-record/. They are environment records, not a
portable dependency lock.

Set these paths before submitting COMET jobs:

```bash
export COMET_PYTHON=/mnt/scratch/t/teng/comet-work/venv/bin/python
export HF_HOME=/mnt/scratch/t/teng/comet-work/huggingface
```

These are Jake's paths. Other users must supply their own accessible environment
and cache paths.

The scratch environment was created with Python's venv module using the Python
installation in ~/miniconda3/envs/mt-comet. Retain that base installation:
the venv depends on it.

For a new setup, create an isolated environment with Python 3.11 and install
unbabel-comet==2.2.7. This alone does not pin its dependencies to the verified
versions. Review the package inventory and validate imports on a compute node
before scoring. Installation must follow the cluster's resource policies.

The tested login shell had a 1 GiB virtual-memory limit. PyTorch import failed
there but succeeded in an allocated GPU job. A login-node import failure does
not by itself establish that the installation is broken.

## Prepare the COMET model cache

Prepare downloads on a node with internet access. These commands download
files without importing PyTorch.

```bash
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

"$COMET_PYTHON" - <<'PYDOWNLOAD'
from huggingface_hub import snapshot_download

snapshot_download(
    "Unbabel/wmt22-comet-da",
    revision="2760a223ac957f30acfb18c8aa649b01cf1d75f2",
    max_workers=1,
)
snapshot_download(
    "xlm-roberta-large",
    revision="c23d21b0620b635a76227c604d44e43a9f0ee389",
    allow_patterns=[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "added_tokens.json",
    ],
    max_workers=1,
)
PYDOWNLOAD
```

These revisions identify the model files used during verification.
The scoring script loads by model name from the cache; it does not explicitly
enforce revision IDs. Preserve the prepared cache, including its main revision
references, when reproducing the offline workflow. Revision-pinned downloads
alone may not create a main cache reference needed by name-based loading.

Disabling Xet avoided a download memory-allocation failure on the login node.

## Submit COMET scoring

COMET scores saved translations without loading the translation checkpoint.

```bash
sbatch jobs/score_comet.sbatch \
  --predictions outputs/eval/smoke-dev20/predictions.jsonl \
  --output outputs/eval/smoke-dev20/comet-check1.json \
  --batch_size 8 \
  --gpus 1
```

Replace the example paths with actual prediction and new output paths.
Read logs/slurm/mt-comet-JOBID.out and check the job status:

```bash
sacct -j JOBID --format=JobID,State,ExitCode
```

The job uses the prepared cache offline and creates temporary files under
the compute node's /tmp. Its cleanup removes the temporary directory only
if empty. A retained-directory message does not itself mean scoring failed.

Using node-local temporary storage avoided the NFS worker-cleanup tracebacks
seen when temporary files were placed on shared scratch.

The scorer uses Unbabel/wmt22-comet-da and reports raw scores, not scores
multiplied by 100. Output includes the system score, sentence scores with IDs,
prediction-file hash, package version, and evaluated row count.

Validate the output:
- Counts and ordered IDs must match the prediction file.
- The prediction hash must match the file scored.
- All scores must be finite.
- The system score must match the mean sentence score within numerical tolerance.

API reference: https://github.com/Unbabel/COMET#scoring-within-python

## Verified results

Cluster verification was completed on 6-7 November 2026 using
runs_test/joint_n1000_seed0/best and alma_dev.jsonl.

These are workflow checks using a 20-step smoke checkpoint.
They are not final experimental results.

| Evaluation | Examples | BLEU | chrF | COMET |
|---|---:|---:|---:|---:|
| Development prefix | 20 | 18.397521 | 46.986086 | 0.794720 |
| Full development | 1002 | 14.276019 | 44.834834 | 0.779338 |

Verification jobs:
- Translation: 1085289 (20 examples), 1085296 (full development).
- COMET: 1089810 (20 examples), 1089815 (full development).
- Reusable COMET job: 1089880 (20 examples).

All three COMET jobs reported COMPLETED with exit code 0:0.
The reusable job's score agreed with the earlier 20-example score to six
decimal places.

Checks passed:
- Six lightweight input/output tests.
- Unique IDs, counts, ordering, and source/reference alignment.
- No empty translations in the checked prediction files.
- Independent full-development BLEU and chrF recalculation.
- Prediction hashes, finite COMET values, and mean/system-score agreement.

Full-development BLEU differed from the training record (14.255027) and
October evaluation (14.256642). The recorded data hash, decoding settings,
and listed package versions matched the October evaluation. The cause of the
small difference has not been established; bitwise reproducibility is not claimed.

## Final evaluation

Use the team's confirmed final checkpoints after settings are agreed.
Evaluate both official test sets:

- data/processed/wmt22_test.jsonl: 1875 examples.
- data/processed/tatoeba_test.jsonl: 469 examples.

WMT22 headline scoring uses Annotator A, stored in en. Do not replace it with
en_ref_b. The supplementary Tatoeba Challenge set does not replace either
official test set.

Example; replace the checkpoint with a real completed run:

```bash
sbatch jobs/evaluate.sbatch \
  --model runs/joint_n1000_seed0/best \
  --data data/processed/wmt22_test.jsonl \
  --num_beams 1 \
  --out_dir outputs/eval/joint-n1000-seed0-wmt22-greedy-v1
```

Omit --limit for final evaluation. Use the same decoding settings across
compared models. Use development data to choose changes, not test scores.
For the unchanged baseline, use --model facebook/m2m100_418M.

Keep checkpoint directories immutable and retain run_info.json. A checkpoint
path alone is not a content fingerprint.

## Remaining work

- Confirm final checkpoint inventory and access.
- Coordinate existing evaluations and decoding comparisons with the team.
- Evaluate available conditions, sizes, and seeds on both official test sets.
- Aggregate results across seeds.
- Compute length/rarity breakdowns using data_utils.assign_bins(), joined by ID.
- Conduct statistical comparisons and qualitative error analysis.

The current scripts do not implement multi-seed aggregation, difficulty-group
breakdowns, or significance testing.

## Sharing the work

Commit the evaluation scripts, both job files, tests, and this guide.
Review and copy selected small result summaries and environment records into
a team-agreed tracked directory if they should be shared through Git.

Do not include model weights, the supplied ZIP, or unrelated files.
Document external contributions and development assistance according to the
course's applicable policy.
