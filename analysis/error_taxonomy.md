# Error taxonomy (draft v0)

Drafted from the **pretrained M2M-100 baseline** on WMT22 and Tatoeba (greedy), using
`analysis/report/*/worst.csv` and `annotation_sample.csv`. Revise it once fine-tuned outputs exist,
because fine-tuning changes which errors dominate. Examples are real outputs: source → reference → system.

## How to annotate
- Run `scripts/analyze_errors.py` on the scored outputs for all systems to create
  `analysis/report/<test_set>/annotation_pairs.csv`.
- This is a long-form sheet: one row per sentence and exact model run. The same sampled sentences
  are used for every run. Even if the analysis command uses a seed glob, human labels are attached
  to each exact run's actual translation, never to seed-averaged scores.
- Fill each row's `error_type`, `severity`, and optional `notes` independently.
  Separate multiple error codes with `;`. Use `OK` or `ALT` alone with severity 0;
  use error codes with severity 1-3. Leave both error_type and severity blank if not yet judged.
- Judge **meaning first, wording second**. The reference is one valid translation, not the only one.
- Judge every system's output for the same sentence before moving on. Hide the system column
  when possible to avoid knowing which sampling condition produced it.
- Re-running sample generation preserves filled labels when sentence, run, output and grouping
  are unchanged. If annotated rows would change or disappear, the script stops: use a different
  `--out_dir` for a different sample. Existing legacy `annotation_sample.csv` files are left intact;
  use the new long-form sheet for paired summaries.

## Compare before and after

Run all commands from the repository root. First score each model with `score_sentences.py`.
Then generate one shared random sample, for example:

```bash
python scripts/analyze_errors.py --test_set wmt22 \
  --baseline uniform_n5000_seed0 \
  --systems pretrained uniform_n5000_seed0 length_n5000_seed0 joint_n5000_seed0
```

After filling `annotation_pairs.csv`, compare the sampling methods against uniform:

```bash
python scripts/compare_annotations.py \
  --annotations analysis/report/wmt22/annotation_pairs.csv \
  --baseline uniform_n5000_seed0 \
  --systems length_n5000_seed0 joint_n5000_seed0
```

To compare the complete fine-tuning pipeline against the unchanged model, use `--baseline pretrained`
with the trained runs as `--systems`. Use exact run names, not globs, in `compare_annotations.py`.
To isolate sampling effects, compare runs with the same training subset size, seed and training budget.

Outputs go to `comparison_<baseline>/` beside the annotation file (or `--out_dir`):
- `paired_errors.csv`: before/after translations, labels, notes, severity and transition for each judged pair.
- `error_transitions.csv`: fixed, introduced, persistent-error and both-correct counts, overall and by
  length, rarity and joint cell; severity changes and paired-annotation coverage are also reported.
- `error_types.csv`: before/after counts for each error type, including types removed or introduced
  even when the sentence still has some other error.

A sentence is "fixed" only when all annotated errors disappear; "persistent_error" means both
outputs have an error, possibly of different types. OK and ALT are both acceptable. Blank labels
are excluded, never assumed correct. Only sentence pairs judged on both sides enter the counts.
Because sampling takes up to 8 sentences per difficulty cell, overall sample counts are **not**
full-test error rates. Use within-cell comparisons, and report sample sizes. Lowest-scoring examples
and largest score disagreements are for illustration, not error-rate estimates.

## Severity
| Score | Meaning |
|---|---|
| 0 | Acceptable. Meaning preserved, even if it differs from the reference |
| 1 | Minor. Meaning preserved, but awkward, ungrammatical or slightly off |
| 2 | Major. Part of the meaning is wrong or missing |
| 3 | Critical. Unusable, wrong overall, or hallucinated |

## Error codes
| Code | Definition | Example |
|---|---|---|
| `REP` | Degenerate repetition: a word or phrase loops | 文摄视觉美学摄影馆优选品质店 回复 → Wenshe Visual Aesthetics Photography Studio Premium Store reply → `Photo Photo Photo Photo ...` |
| `OMIT` | Content from the source is dropped | 不满意也不行啦！运费都要 20 元 → I can only be satisfied! Even the delivery fee is 20 yuan. → `I am not happy! $20` (delivery fee lost) |
| `ADD` | Content that is not in the source is added (not a loop) | |
| `SENSE` | Wrong word sense, idiom taken literally, or wrong relation or polarity | 装病是没用的 → There is no point in pretending to be sick. → `Disease is useless.` ; 有火吗？ → Have you got a light? → `Is there fire?` |
| `NE` | Named entity or proper noun wrong, mangled or invented | 日昌外卖。 → Richang takeout. → `Day of sale.` ; `Cummit for Democracy` |
| `TERM` | Domain term or specialised vocabulary wrong | 检查井砌块砖机设备 → inspection shaft module brick equipment → `grid cluster equipment` ; 强精 (TCM) translated as `strong ejaculation` |
| `NUM` | Number, unit, date or currency wrong or lost | 运费都要 20 元 → 20 yuan → `$20` (currency changed) |
| `FLU` | Grammar or fluency problem, meaning still recoverable | 可以吗 → Is it Okay? → `can?` ; 早！ → Morning! → `early to!` |
| `UNTR` | Source text left untranslated (the automatic flag `has_cjk` finds some) | |
| `ALT` | **Valid alternative**: acceptable translation that the reference-based metrics punish (use with severity 0) | 别碰！ → Hands off! → `Don't touch it!` ; 值得！ → Deserving! → `It is worth it!` |

Notes on the codes:
- Short, context-free fragments (`好的`, `可以吗`) are ambiguous. If several translations are fine, use `ALT`, and do not count it as a model error.
- `ALT` matters: a low chrF does not always mean a bad translation. Report its share per bin and per system, so the metric
  scores are not read as pure quality differences.
- Keep `REP` separate from `ADD`. Repetition is a decoding failure, and the automatic flag `repeated` covers part of it.

## What the baseline shows so far
Observations from the lowest-scoring sentences, not conclusions:
- The lowest scores are mostly **short, context-free inputs** (product reviews and chat fragments in WMT22, one-word commands in Tatoeba).
  On WMT22 the `short` bin has the lowest mean chrF (38.4) and `long` the highest (48.0). This is the opposite of what a
  naive "longer is harder" assumption predicts, so check it against the proposal's expectation about WMT22.
- **Repetition loops** appear in noisy, list-like or very long inputs (2.7% of WMT22 outputs are flagged `repeated`).
- Names, brand names and domain terms (`NE`, `TERM`) are a main source of major errors in `rare` cells.

## Questions the sheet should answer
The proposal expects length balancing to help on WMT22 and possibly hurt on Tatoeba, with the benefit shrinking as the
training set grows. For each system pair, count by bin:
1. Does `REP` or `OMIT` on long sentences go down under `length` and `joint` sampling?
2. Do `SENSE` and `FLU` errors on short sentences go up (the possible harm on Tatoeba)?
3. Do `NE` and `TERM` errors in `rare` cells go down under `joint` sampling?
4. What share of low-chrF sentences are `ALT`?

## Agreement (optional, cheap)
Have two people annotate the same 30 sentences and report Cohen's kappa on the main code. If agreement is low, tighten the definitions above.
