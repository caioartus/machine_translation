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
    df = runs[0].copy()
    numeric = [c for c in SCORES + ["len_ratio", "missing_nums"] if c in df]
    for col in numeric:
        df[col] = np.mean([r[col].astype(float) for r in runs], axis=0)
    for col in FLAGS:  # share of the runs in which the flag was raised
        df[col] = np.mean([r[col].astype(str).eq("True") for r in runs], axis=0)
    df.attrs["n_runs"] = len(runs)
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
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


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
    args = parser.parse_args()
    if args.baseline not in args.systems:
        raise SystemExit("--baseline must also be listed in --systems")

    systems = {spec: load_system(spec, args.test_set, args.scored_dir) for spec in args.systems}
    first = next(iter(systems.values()))
    for name, df in systems.items():
        if df["id"].tolist() != first["id"].tolist():
            raise SystemExit(f"{name} was not scored on the same sentences as the other systems")

    out = Path(args.out_dir) / args.test_set
    out.mkdir(parents=True, exist_ok=True)

    table = summarise(systems)
    table.to_csv(out / "groups.csv", index=False, float_format="%.4f")
    diffs = deltas(systems, args.baseline, args.n_boot, args.seed)
    diffs.to_csv(out / "deltas.csv", index=False, float_format="%.4f")

    # Lowest-chrF sentences per system.
    worst = []
    for name, df in systems.items():
        low = df.nsmallest(args.k, "chrf")[TEXT_COLS[:4] + ["cell", "chrf", "len_ratio"]].copy()
        low.insert(0, "system", name)
        worst.append(low)
    pd.concat(worst).to_csv(out / "worst.csv", index=False, float_format="%.4f")

    # Largest differences from the baseline, in either direction.
    base = systems[args.baseline]
    disagree = []
    for name, df in systems.items():
        if name == args.baseline:
            continue
        gap = df["chrf"] - base["chrf"]
        top = df.loc[gap.abs().nlargest(args.k).index, TEXT_COLS[:4] + ["cell"]].copy()
        top.insert(0, "system", name)
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
        sample[f"hyp[{name}]"] = df.loc[sample.index, "hyp"]
    for col in ["error_type", "severity", "notes"]:
        sample[col] = ""
    sample.to_csv(out / "annotation_sample.csv", index=False)

    # Readable report.
    lines = [f"# Error analysis: {args.test_set}", "",
             f"Baseline for differences: `{args.baseline}`. Runs per system: "
             + ", ".join(f"`{n}` {d.attrs['n_runs']}" for n, d in systems.items()) + ".", "",
             "## Mean sentence chrF by group", ""]
    pivot = table.pivot_table(index=["group_type", "group", "n"], columns="system", values="chrf", sort=False)
    lines += [md_table(pivot[list(systems)].reset_index()), "",
              "## Difference from baseline (chrF, 95% paired bootstrap over sentences)", ""]
    if not diffs.empty:
        show = diffs.assign(ci=lambda d: d.apply(lambda r: f"[{r.ci_low:+.2f}, {r.ci_high:+.2f}]", axis=1),
                            sig=lambda d: d["excludes_zero"].map({True: "yes", False: ""}))
        lines += [md_table(show[["system", "group_type", "group", "n", "delta_chrf", "ci", "sig"]]), ""]
    overall = table[table["group_type"] == "overall"]
    lines += ["## Error flags (share of sentences)", "",
              f"too_short: output under {SHORT_RATIO} of the reference length in words; too_long: over {LONG_RATIO}.", "",
              md_table(overall[["system", "n", "too_short", "too_long", "empty", "has_cjk", "repeated", "missing_nums"]],
                       floatfmt="{:.3f}"), ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}/ (groups, deltas, worst, disagreements, annotation_sample, report.md)")


if __name__ == "__main__":
    main()
