"""Error analysis, step 2: compare systems by difficulty group and pick sentences to read.

Reads the per-sentence files made by score_sentences.py (analysis/scored/<system>/<test_set>.csv)
and writes, to analysis/report/<test_set>/:
    groups.csv             per system and group (overall, length bin, rarity bin, cell): n, mean
                           chrF, sentence BLEU, COMET if present, and the rate of each error flag
    deltas.csv             each system minus the baseline, per group: mean chrF difference with a
                           95% paired-bootstrap interval over sentences
    worst.csv              the k lowest-chrF sentences of each system
    disagreements.csv      the k sentences where each system differs most from the baseline
    annotation_sample.csv  a stratified random sample, same sentences for every system, with empty
                           error_type / severity / notes columns for manual annotation
    annotation_pairs.csv   one row per sampled sentence and exact run, with independent labels;
                           annotate this file, then run compare_annotations.py
    report.md              the tables above in readable form

A system can be given as a glob over seeds, e.g. "uniform_n5000_seed*". Per-sentence scores are
then averaged over the matching runs, and the translations shown are those of the first run.
The bootstrap resamples sentences only, so the intervals do not include seed-to-seed variance:
read them together with the spread over seeds in the official corpus-level scores.

Run from the repository root, after score_sentences.py for every system:
    python scripts/analyze_errors.py --test_set wmt22 --baseline "uniform_n5000_seed*" \
        --systems "uniform_n5000_seed*" "length_n5000_seed*" "joint_n5000_seed*" pretrained
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import all_cells, LENGTH_LABELS, RARITY_LABELS
from error_analysis_utils import align_to, paired_interval
from quantitative_errors import write_quantitative

SCORES = ["chrf", "bleu", "comet"]
FLAGS = ["empty", "has_cjk", "repeated"]
# Thresholds for "much shorter / longer than the reference" (words in output / words in reference).
SHORT_RATIO, LONG_RATIO = 0.6, 1.6
TEXT_COLS = ["id", "zh", "ref", "hyp", "src_len", "tgt_len", "rarity_mean", "length_bin", "rarity_bin", "cell"]


def load_system(spec, test_set, scored_dir):
    """Load (and average over seeds) the scored sentences of one system spec."""
    paths = sorted(Path(scored_dir).glob(f"{spec}/{test_set}.csv"))
    if not paths:
        raise SystemExit(f"no scored files match {scored_dir}/{spec}/{test_set}.csv")
    runs = [pd.read_csv(p, keep_default_na=False) for p in paths]
    runs = [align_to(r, runs[0]["id"].tolist(), str(path)) for r, path in zip(runs, paths)]
    for path, run in zip(paths, runs):
        if run["id"].duplicated().any():
            raise SystemExit(f"duplicate sentence ids in {path}")
        for col in ["id", "zh", "ref", "src_len", "tgt_len", "rarity_mean", "length_bin", "rarity_bin", "cell"]:
            if run[col].tolist() != runs[0][col].tolist():
                raise SystemExit(f"mismatched {col} across runs matching {spec}")
    df = runs[0].copy()
    numeric = [c for c in SCORES + ["len_ratio", "missing_nums"] if c in df]
    for col in numeric:
        df[col] = np.mean([r[col].astype(float) for r in runs], axis=0)
    for col in FLAGS:  # share of the runs in which the flag was raised
        df[col] = np.mean([r[col].astype(str).eq("True") for r in runs], axis=0)
    df.attrs["n_runs"] = len(runs)
    df.attrs["representative_run"] = paths[0].parent.name
    df.attrs["representative_data"] = runs[0]
    return df


def groups_of(df):
    """Yield (group_type, group, rows) for overall, length bins, rarity bins and the 8 cells."""
    yield "overall", "all", df
    for label in LENGTH_LABELS:
        yield "length", label, df[df["length_bin"] == label]
    for label in RARITY_LABELS:
        yield "rarity", label, df[df["rarity_bin"] == label]
    for cell in all_cells():
        yield "cell", cell, df[df["cell"] == cell]


def summarise(systems):
    records = []
    for name, df in systems.items():
        for gtype, group, rows in groups_of(df):
            if rows.empty:
                continue
            rec = {"system": name, "group_type": gtype, "group": group, "n": len(rows)}
            for col in SCORES:
                if col in rows:
                    rec[col] = rows[col].mean()
            rec["too_short"] = (rows["len_ratio"] < SHORT_RATIO).mean()
            rec["too_long"] = (rows["len_ratio"] > LONG_RATIO).mean()
            for col in FLAGS:
                rec[col] = rows[col].mean()
            rec["missing_nums"] = (rows["missing_nums"] > 0).mean()
            records.append(rec)
    return pd.DataFrame(records)


def bootstrap_ci(diff, n_boot, rng):
    """95% interval of the mean of `diff` (paired differences), resampling sentences."""
    return paired_interval(diff, n_boot, int(rng.integers(0, 2**31)))


def deltas(systems, baseline, n_boot, seed):
    rng = np.random.default_rng(seed)
    base = systems[baseline]
    records = []
    for name, df in systems.items():
        if name == baseline:
            continue
        for gtype, group, rows in groups_of(df):
            if rows.empty:
                continue
            diff = rows["chrf"].to_numpy() - base.loc[rows.index, "chrf"].to_numpy()
            lo, hi = bootstrap_ci(diff, n_boot, rng)
            records.append({"system": name, "baseline": baseline, "group_type": gtype, "group": group,
                            "n": len(rows), "delta_chrf": diff.mean(), "ci_low": lo, "ci_high": hi,
                            "excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(records)


def write_annotation_pairs(systems, picked_ids, test_set, scored_dir, path):
    """Keep labels attached to exact outputs, never to averaged seed scores."""
    rows, seen = [], set()
    for spec in systems:
        for source in sorted(Path(scored_dir).glob(f"{spec}/{test_set}.csv")):
            if source in seen:
                continue
            seen.add(source)
            run = pd.read_csv(source, keep_default_na=False)
            for row in run[run["id"].isin(picked_ids)].to_dict("records"):
                rows.append({**{k: row[k] for k in ["id", "zh", "ref", "hyp", "length_bin", "rarity_bin", "cell"]},
                             "system": source.parent.name, "error_type": "", "severity": "", "notes": ""})
    fresh = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False, dtype=str)
        keys = ["id", "system"]
        if old.duplicated(keys).any():
            raise SystemExit(f"Duplicate labels in {path}; resolve them before regenerating")
        existing = fresh.set_index(keys)
        for row in old.to_dict("records"):
            key = (row["id"], row["system"])
            if not any(str(row.get(c, "")).strip() for c in ["error_type", "severity", "notes"]):
                continue
            if key not in existing.index or any(str(existing.loc[key, c]) != str(row[c]) for c in ["zh", "ref", "hyp", "length_bin", "rarity_bin", "cell"]):
                raise SystemExit(f"Annotated output changed or left the sample: {key}. Use a different --out_dir to preserve {path}")
            for c in ["error_type", "severity", "notes"]:
                existing.loc[key, c] = row.get(c, "")
        fresh = existing.reset_index()
    fresh.to_csv(path, index=False)


def md_table(df, floatfmt="{:.2f}"):
    """A GitHub markdown table (no extra dependency)."""
    def cell(v):
        return floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--test_set", required=True)
    parser.add_argument("--systems", nargs="+", required=True)
    parser.add_argument("--baseline", required=True, help="one of the systems; deltas are relative to it")
    parser.add_argument("--scored_dir", default="analysis/scored")
    parser.add_argument("--out_dir", default="analysis/report")
    parser.add_argument("--k", type=int, default=20, help="sentences listed per system in worst/disagreements")
    parser.add_argument("--sample_per_cell", type=int, default=8, help="annotation sample size per cell")
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--comparison", choices=["sampling", "decoding"], default="sampling",
                        help="sampling matches seeds/size/beams; decoding matches seeds/size/condition")
    args = parser.parse_args()
    if args.baseline not in args.systems:
        raise SystemExit("--baseline must also be listed in --systems")
    if args.n_boot < 1 or args.sample_per_cell < 1 or args.k < 1:
        parser.error("n_boot, sample_per_cell and k must be positive")

    systems = {spec: load_system(spec, args.test_set, args.scored_dir) for spec in args.systems}
    first = next(iter(systems.values()))
    for name, df in systems.items():
        if df["id"].tolist() != first["id"].tolist():
            raise SystemExit(f"{name} was not scored on the same sentences as the other systems")

    out = Path(args.out_dir) / args.test_set
    out.mkdir(parents=True, exist_ok=True)
    quantitative = write_quantitative(args.systems, args.baseline, args.test_set,
                                     args.scored_dir, out, args.n_boot, args.seed, args.comparison)

    table = summarise(systems)
    table.to_csv(out / "groups.csv", index=False, float_format="%.4f")
    diffs = deltas(systems, args.baseline, args.n_boot, args.seed)
    diffs.to_csv(out / "deltas.csv", index=False, float_format="%.4f")

    # Lowest-chrF sentences per system.
    worst = []
    for name, df in systems.items():
        actual = df.attrs["representative_data"]
        low = actual.nsmallest(args.k, "chrf")[TEXT_COLS[:4] + ["cell", "chrf", "len_ratio"]].copy()
        low.insert(0, "system", df.attrs["representative_run"])
        low.insert(0, "system_spec", name)
        worst.append(low)
    pd.concat(worst).to_csv(out / "worst.csv", index=False, float_format="%.4f")

    # Largest differences from the baseline, in either direction.
    base = systems[args.baseline]
    base_run = base.attrs["representative_run"]
    base = base.attrs["representative_data"]
    disagree = []
    for name, df in systems.items():
        if name == args.baseline:
            continue
        actual_name = df.attrs["representative_run"]
        df = df.attrs["representative_data"]
        gap = df["chrf"] - base["chrf"]
        top = df.loc[gap.abs().nlargest(args.k).index, TEXT_COLS[:4] + ["cell"]].copy()
        top.insert(0, "system", actual_name)
        top.insert(0, "baseline_run", base_run)
        top = top.rename(columns={"hyp": "hyp_system"})
        top["hyp_baseline"] = base.loc[top.index, "hyp"]
        top["chrf_system"] = df.loc[top.index, "chrf"]
        top["chrf_baseline"] = base.loc[top.index, "chrf"]
        top["delta_chrf"] = gap.loc[top.index]
        disagree.append(top)
    if disagree:
        pd.concat(disagree).to_csv(out / "disagreements.csv", index=False, float_format="%.4f")

    # Stratified annotation sample: the same sentences for every system, so errors can be compared.
    rng = np.random.default_rng(args.seed)
    picked = []
    for cell in all_cells():
        ids = first.index[first["cell"] == cell].to_numpy()
        picked.extend(rng.choice(ids, size=min(args.sample_per_cell, len(ids)), replace=False))
    sample = first.loc[sorted(picked), ["id", "cell", "zh", "ref"]].copy()
    for name, df in systems.items():
        sample[f"hyp[{df.attrs['representative_run']}]"] = df.attrs["representative_data"].loc[sample.index, "hyp"]
    for col in ["error_type", "severity", "notes"]:
        sample[col] = ""
    # Keep the old wide sheet for compatibility; never overwrite existing manual work.
    legacy_path = out / "annotation_sample.csv"
    if not legacy_path.exists():
        sample.to_csv(legacy_path, index=False)
    write_annotation_pairs(systems, sample["id"].tolist(), args.test_set, args.scored_dir,
                           out / "annotation_pairs.csv")
    blind_path = out / "annotation_blind.csv"
    if not blind_path.exists():
        pairs = pd.read_csv(out / "annotation_pairs.csv", keep_default_na=False)
        names = sorted(pairs.system.unique())
        shuffled = np.random.default_rng(args.seed).permutation(names)
        mapping = {name: f"S{i+1:02d}" for i, name in enumerate(shuffled)}
        keys = pairs[["id", "system"]].copy()
        keys["sample_id"] = [f"A{i+1:05d}" for i in range(len(keys))]
        keys["system_code"] = pairs.system.map(mapping)
        keys.to_csv(out / "annotation_key.csv", index=False)
        pairs["sample_id"] = keys.sample_id
        pairs["system_code"] = keys.system_code
        pairs = pairs.drop(columns="system").sort_values(["id", "system_code"])
        pairs.to_csv(blind_path, index=False)

    # Readable report.
    lines = [f"# Error analysis: {args.test_set}", "",
             f"Baseline for differences: `{args.baseline}`. Runs per system: "
             + ", ".join(f"`{n}` {d.attrs['n_runs']}" for n, d in systems.items()) + ".", "",
             "## Quantitative results", "",
             "`groups_by_run.csv` recomputes corpus BLEU/chrF within each exact run and group. "
             "`groups_across_seeds.csv` reports means and sample SD across runs. "
             "`paired_changes_by_run.csv` reports matched sentence-score changes and win/loss/tie shares. "
             "`coverage_score_correlations.csv` describes associations within length groups. "
             "`paired_examples_by_run.csv` keeps actual text and exact scores together.", "",
             "Bootstrap intervals are exploratory, unadjusted and sentence-only: they do not model "
             "seed or document dependence and are not declarations of significance. "
             "Corpus differences and coverage correlations do not establish causality.", "",
             ""]
    overall_corpus = quantitative[quantitative.group_type == "overall"]
    lines += ["## Corpus metrics across seeds (mean and sample SD)", "",
              md_table(overall_corpus[["system_spec", "n_runs", "n_sentences", "corpus_bleu_mean", "corpus_bleu_std", "corpus_chrf_mean", "corpus_chrf_std"]]), ""]
    lines += ["## Mean sentence chrF by group (not corpus chrF)", ""]
    pivot = table.pivot_table(index=["group_type", "group", "n"], columns="system", values="chrf", sort=False)
    lines += [md_table(pivot[list(systems)].reset_index()), "",
              "## Difference from baseline (chrF, 95% paired bootstrap over sentences)", ""]
    if not diffs.empty:
        show = diffs.assign(ci=lambda d: d.apply(lambda r: f"[{r.ci_low:+.2f}, {r.ci_high:+.2f}]", axis=1),
                            sig=lambda d: d["excludes_zero"].map({True: "yes", False: ""}))
        lines += [md_table(show[["system", "group_type", "group", "n", "delta_chrf", "ci", "sig"]]), ""]
    overall = table[table["group_type"] == "overall"]
    lines += ["## Candidate error flags (not human-confirmed errors)", "",
              f"too_short: output under {SHORT_RATIO} of the reference length in words; too_long: over {LONG_RATIO}.", "",
              md_table(overall[["system", "n", "too_short", "too_long", "empty", "has_cjk", "repeated", "missing_nums"]],
                       floatfmt="{:.3f}"), ""]
    lines += ["## Paired human error analysis", "",
              "Annotate `annotation_pairs.csv`: each exact run has its own row and labels. "
              "Use OK or ALT with severity 0, or error codes with severity 1–3. "
              "Then run `scripts/compare_annotations.py` with an exact baseline run and comparison runs.", "",
              "Blank annotations are excluded, not counted as correct. "
              "The sample is balanced across difficulty cells; its overall counts are not population error rates. "
              "Do not use the lowest-scoring examples to estimate error rates.", ""]
    lines += ["Use `annotation_blind.csv` for anonymous annotation and keep `annotation_key.csv` "
              "separate from annotators. compare_annotations.py --key restores run identities. "
              "Existing annotation sheets/mappings are never silently replaced. "
              "Legacy worst/disagreements show a named representative run, not averaged seed scores; "
              "prefer paired_examples_by_run.csv for matched-seed examples.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(quantitative[quantitative.group_type == "overall"].to_string(index=False))
    print(f"wrote {out}/ (corpus metrics, paired changes, coverage associations, annotations and legacy reports)")


if __name__ == "__main__":
    main()
