# Quantitative error analysis

Run commands from the repository root in the existing project environment. These
tools use CPU tokenization and existing translations; they never fine-tune a model
or invoke an external scoring service. The source tokenizer must be cached or
downloadable. COMET scoring and translation generation remain the evaluation
pipeline's responsibility.

The questions are: (1) quality by length and ALMA-relative rarity; (2) which real
errors underlie low scores; (3) whether training/test coverage differences are
associated with score differences. Corpus identity is not a domain label. ALMA
coverage does not measure the pretrained model's complete vocabulary exposure.

## 1. Corpus profiles

```bash
python scripts/compare_corpus_profiles.py \
  --out_dir analysis/quantitative_v1/corpus_profiles
```

This reads cleaned ALMA train/dev and the official WMT22/Tatoeba processed text,
the existing difficulty CSVs and `subword_freq.tsv`. It verifies that retokenizing
ALMA reproduces the saved frequency table before computing new features.
Choose a fresh output directory on a rerun; existing results are not overwritten.

Outputs:

- `corpus_summary.csv`: length/rarity summaries, token-weighted unseen and low-frequency shares.
- `<split>.csv`: existing bins plus coverage features, by sentence ID.
- `group_distribution.csv`: counts/shares for fixed bins, cells, and available domains.
- `top_subwords.csv`: top-100 candidates per split plus top-100 ALMA candidates,
  raw counts and frequency per 10,000 tokens (not an exhaustive vocabulary table).
- `length_distribution.png`, `rarity_distribution.png`: empirical CDFs.
- `provenance.json`: thresholds and input SHA256 hashes.

Unseen means count zero in ALMA. Low-frequency means count 1 through 5 by default,
excluding unseen. All datasets use the same tokenizer and training reference.
Training sentences subtract their own subwords (leave-one-out), matching the
existing training rarity definition; held-out text uses full training counts.
Consequently, training and held-out reference pools differ slightly; interpret
training coverage as leave-one-out coverage, not ordinary in-sample OOV.
Punctuation is included; these are subword statistics, not segmented word counts.
Top words cannot establish news/conversation domains. Do not redefine rarity
using test frequencies or alter the sampling bins after inspecting test results.

## 2. Score exact outputs

Original plain-text files still work (line order is assumed, not verifiable):

```bash
python scripts/score_sentences.py --system pretrained --test_set wmt22 \
  --num_beams 1 --out_dir analysis/quantitative_v1/scored
python scripts/score_sentences.py --system pretrained --test_set tatoeba \
  --num_beams 1 --out_dir analysis/quantitative_v1/scored
```

For fine-tuned outputs use explicit, distinct system names and metadata:

```bash
python scripts/score_sentences.py \
  --system uniform_nfull_seed0_beam4 --test_set wmt22 \
  --hyp outputs/uniform_nfull_seed0/wmt22.txt \
  --condition uniform --train_size full --seed 0 --num_beams 4 \
  --out_dir analysis/quantitative_v1/scored
```

The hypotheses must actually have been generated with the stated model/beam.
Metadata labels do not change decoding. Prefer JSONL with `id`, `hyp` (and optional
`zh`, `ref`), which is reordered and validated against the source IDs. Optional
`generation_limit_hit` must be a JSON boolean recorded by the generator, not inferred.
Duplicate, missing or extra IDs and mismatched source/reference text are rejected.
Use `--comet file.txt` only for one finite COMET score per sentence in processed
source order; this does not run COMET and cannot verify the IDs in a plain score file.

The development JSONL from compare_decoding.py can be used without treating it as
test output:

```bash
python scripts/score_sentences.py \
  --system uniform_nfull_seed0_beam4 --test_set alma_dev --split alma_dev \
  --hyp runs_decode_ablation/1083476/uniform_nfull_seed0/beam4.jsonl \
  --condition uniform --train_size full --seed 0 --num_beams 4 \
  --out_dir analysis/dev_decoding/scored
```

Replace the batch/path with the file's actual location. Scored outputs get input
hashes in `<test_set>.meta.json`. Existing scored files require `--overwrite` or
a new output root. Input lengths beyond training max_length and output lengths
near it are screening flags, not proof of truncation or omission.

## 3. Sampling comparison

After scoring all corresponding systems (the glob matches seeds only):

```bash
python scripts/analyze_errors.py --test_set wmt22 \
  --scored_dir analysis/quantitative_v1/scored \
  --out_dir analysis/quantitative_v1/report \
  --baseline 'uniform_nfull_seed*_beam4' \
  --systems 'uniform_nfull_seed*_beam4' 'length_nfull_seed*_beam4' 'joint_nfull_seed*_beam4'
```

Repeat for Tatoeba. Compare the same training size, cap, learning rate, budget,
seed and decoding configuration. Keep a separate scored/report root for cap/LR
ablations; not all training settings can be recovered from a system name.
Unknown seed/beam labels are not evidence that settings match. Populate metadata
from actual run records. Mixed conditions, sizes or beams within one glob are rejected.
Use `--baseline pretrained --systems pretrained` to generate a descriptive
baseline report when trained translations are unavailable; this produces no method gains.

New report files:

- `groups_by_run.csv`: corpus BLEU/chrF recomputed from real text in each group,
  mean sentence chrF/COMET, candidate-flag shares, domains if available.
- `groups_across_seeds.csv`: metric means and sample SD across exact runs, sentence
  counts and number of runs. No pooling of seed-sentence rows as independent samples.
- `paired_changes_by_run.csv`: seed-matched sentence chrF/COMET changes, medians,
  win/loss/tie shares, 10th/90th percentiles, exploratory sentence-bootstrap intervals.
- `paired_changes_across_seeds.csv`: mean gains and SD of per-seed gains.
- `delta_chrf_*.png`: mean paired sentence chrF gains over fixed length/rarity cells.
- `coverage_score_correlations.csv`: Spearman associations overall and within each
  fixed length group; constant variables produce missing coefficients, not zero.
- `paired_examples_by_run.csv`: largest disagreements with exact seed translations
  and corresponding scores. These are illustrative, not representative error-rate samples.

Legacy reports remain available. `groups.csv` contains mean sentence scores, NOT
corpus BLEU/chrF. `worst.csv` and `disagreements.csv` now identify their representative
exact run and never show averaged scores beside a single seed's text. Prefer the
new paired examples for seed-matched comparisons.

Intervals are paired over sentences only: no seed/document uncertainty and no
multiple-comparison correction. Do not call a cell significant merely because its
interval excludes zero. Three seeds give limited precision. For final corpus-level
significance use the evaluation pipeline's paired tests; where possible preserve
WMT document blocks. Coverage correlations do not establish causal domain effects.

## 4. Decoding comparison

Score both beam versions separately, then compare within one training condition:

```bash
python scripts/analyze_errors.py --test_set alma_dev --comparison decoding \
  --scored_dir analysis/dev_decoding/scored --out_dir analysis/dev_decoding/report \
  --baseline 'uniform_nfull_seed*_beam1' \
  --systems 'uniform_nfull_seed*_beam1' 'uniform_nfull_seed*_beam4'
```

Use different output roots for length and joint so reports do not overwrite each
other. Decoding matches condition, training size and seed; sampling matches beam,
size and seed. A model comparison against pretrained is intentionally allowed.
The bootstrap is sentence-score based, not a bootstrap of corpus BLEU.

## 5. Human checks

`annotation_pairs.csv` retains exact-run labels and preserves existing manual work.
`annotation_blind.csv` replaces system names with anonymous codes. Keep
`annotation_key.csv` separate from annotators. Sample up to 8 source sentences in
each of the 8 cells (at most 64 unique sources), reused across outputs. Existing
anonymous sheets/mappings are never overwritten; a new sample/configuration needs
a new output directory. Labels apply separately to each output, not to a sentence
across all systems. Use the existing taxonomy and judge against the Chinese source.

After annotation:

```bash
python scripts/compare_annotations.py \
  --annotations analysis/quantitative_v1/report/wmt22/annotation_blind.csv \
  --key analysis/quantitative_v1/report/wmt22/annotation_key.csv \
  --baseline uniform_nfull_seed0_beam4 \
  --systems length_nfull_seed0_beam4 joint_nfull_seed0_beam4 \
  --population_features data/features/wmt22_test.csv
```

Only completed paired labels enter counts and rates. Blanks are never correct
outputs. Rates in error_types/error_transitions describe the annotation sample.
Optional population estimates poststratify by full-test cell sizes, and are emitted
only if every population cell has a fully judged pair sample. They assume random
within-cell sampling and must not be used for cherry-picked worst cases; they have
no uncertainty interval. Keep all sampled rows, including unfinished ones.
For agreement, ask another annotator to label an independent copy of the same
anonymous sheet; do not overwrite the first person's labels.

## Local validation and current scope

```bash
python -m unittest discover -s tests -v
```

The local `analysis/quantitative_v1/` run uses only available pretrained WMT22 and
Tatoeba translations. It is a baseline workflow demonstration, not evidence of
balanced-sampling gains. Synthetic paired tests validate arithmetic and joins;
their numbers are not research results. Do not tune on final test-set outputs.
