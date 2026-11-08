"""Quantitative reports on exact translations, with paired seeds and fixed bins.

Called by analyze_errors.py; never substitutes averaged seed scores for real text.
Bootstrap intervals describe sentences only, are unadjusted/exploratory, and do
not include seed or document dependence. They are not significance declarations.
"""
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

from data_utils import LENGTH_LABELS, RARITY_LABELS
from error_analysis_utils import (align_to, group_rows, infer_run_metadata,
                                  paired_interval, unique_ids, write_json, file_hash)

SOURCE_COLUMNS = ["zh", "ref", "src_len", "tgt_len", "rarity_mean", "length_bin", "rarity_bin", "cell"]


def constant_metadata(df, name):
    meta = infer_run_metadata(name)
    for key in meta:
        if key in df:
            if df[key].nunique(dropna=False) != 1:
                raise ValueError(f"{name}: mixed {key} values in one scored run")
            meta[key] = str(df[key].iloc[0])
        else:
            meta[key] = str(meta[key])
    return meta


def read_runs(specs, test_set, scored_dir):
    runs, ids, first = {}, None, None
    for spec in specs:
        paths = sorted(Path(scored_dir).glob(f"{spec}/{test_set}.csv"))
        if not paths:
            raise ValueError(f"no scored runs match {spec}/{test_set}")
        entries = []
        for path in paths:
            df = pd.read_csv(path, keep_default_na=False)
            unique_ids(df, str(path))
            ids = df.id.tolist() if ids is None else ids
            df = align_to(df, ids, str(path))
            if first is None:
                first = df
            for col in SOURCE_COLUMNS + [c for c in ["domain", "docid", "unseen_ratio", "lowfreq_ratio", "lowfreq_threshold"] if c in first]:
                if col not in df or df[col].tolist() != first[col].tolist():
                    raise ValueError(f"{path}: incompatible source feature {col}")
            for metric in [c for c in ["chrf", "bleu", "comet"] if c in df]:
                df[metric] = pd.to_numeric(df[metric], errors="raise")
                if not np.isfinite(df[metric]).all():
                    raise ValueError(f"{path}: nonfinite {metric}")
            entries.append((path.parent.name, df, constant_metadata(df, path.parent.name)))
        # A spec averages seeds only, never multiple decoding/training conditions.
        for key in ["condition", "train_size", "num_beams"]:
            if len({e[2][key] for e in entries}) > 1:
                raise ValueError(f"{spec}: mixes {key}; use separate system specs")
        if len({e[2]["seed"] for e in entries}) != len(entries):
            raise ValueError(f"{spec}: duplicate or unknown seeds; supply explicit metadata")
        runs[spec] = entries
    return runs


def corpus_metrics(df):
    bleu, chrf = BLEU(tokenize="13a"), CHRF()
    refs, hyps = [df.ref.tolist()], df.hyp.tolist()
    return {"corpus_bleu": bleu.corpus_score(hyps, refs).score,
            "corpus_chrf": chrf.corpus_score(hyps, refs).score,
            "mean_sentence_chrf": df.chrf.mean(),
            "bleu_signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature())}


def matched_baseline(entries, meta, comparison):
    if len(entries) == 1 and entries[0][2]["condition"] == "pretrained":
        return entries[0]
    keys = ["seed", "train_size", "num_beams"] if comparison == "sampling" else ["seed", "train_size", "condition"]
    matches = [e for e in entries if all(e[2][k] == meta[k] for k in keys)]
    if len(matches) != 1:
        raise ValueError(f"need one baseline matched by {keys}; got {len(matches)} for {meta}")
    if comparison == "decoding" and (meta["num_beams"] == "unknown" or matches[0][2]["num_beams"] == "unknown"):
        raise ValueError("decoding comparisons require explicit num_beams on both sides")
    return matches[0]


def paired_rows(df, base, n_boot, seed, tie_epsilon=0.):
    records = []
    for kind, label, rows in group_rows(df):
        if rows.empty:
            continue
        for metric in [m for m in ["chrf", "comet"] if m in rows and m in base]:
            diff = rows[metric].to_numpy() - base.loc[rows.index, metric].to_numpy()
            lo, hi = paired_interval(diff, n_boot, seed)
            records.append({"group_type": kind, "group": label, "metric": metric, "n": len(rows),
                            "mean_delta": diff.mean(), "median_delta": np.median(diff),
                            "p10_delta": np.quantile(diff, .1), "p90_delta": np.quantile(diff, .9),
                            "win_share": np.mean(diff > tie_epsilon),
                            "loss_share": np.mean(diff < -tie_epsilon),
                            "tie_share": np.mean(np.abs(diff) <= tie_epsilon),
                            "tie_epsilon": tie_epsilon, "ci_low": lo, "ci_high": hi,
                            "ci_scope": "unadjusted_sentence_only_not_seed_or_document"})
    return records


def coverage_correlations(df):
    rows = []
    for kind, label, group in group_rows(df):
        if kind not in ["overall", "length"] or len(group) < 3:
            continue
        for feature in [c for c in ["src_len", "rarity_mean", "unseen_ratio", "lowfreq_ratio"] if c in group]:
            for metric in [m for m in ["chrf", "comet"] if m in group]:
                if group[feature].nunique() < 2 or group[metric].nunique() < 2:
                    corr = None
                else:
                    corr = group[feature].rank().corr(group[metric].rank())
                rows.append({"group_type": kind, "group": label, "feature": feature,
                             "metric": metric, "n": len(group), "spearman_rho": corr})
    return rows


def write_quantitative(specs, baseline, test_set, scored_dir, out, n_boot=2000, seed=0, comparison="sampling"):
    out = Path(out)
    runs = read_runs(specs, test_set, scored_dir)
    metrics, changes, corrs = [], [], []
    examples = []
    for spec, entries in runs.items():
        for name, df, meta in entries:
            identity = {"system_spec": spec, "run": name, **meta}
            for kind, label, group in group_rows(df):
                if group.empty:
                    continue
                rec = {**identity, "group_type": kind, "group": label, "n": len(group), **corpus_metrics(group)}
                if "comet" in group:
                    rec["mean_comet"] = group.comet.mean()
                for flag in [c for c in ["empty", "has_cjk", "repeated", "input_exceeds_max_length",
                                        "output_near_training_limit", "generation_limit_hit"] if c in group]:
                    rec["candidate_" + flag + "_share"] = group[flag].astype(str).str.lower().eq("true").mean()
                metrics.append(rec)
            corrs.extend({**identity, **r} for r in coverage_correlations(df))
            if spec == baseline:
                continue
            bname, base, _ = matched_baseline(runs[baseline], meta, comparison)
            changes.extend({**identity, "baseline_run": bname, **r}
                           for r in paired_rows(df, base, n_boot, seed))
            delta = df.chrf - base.chrf
            for i in delta.abs().nlargest(min(20, len(df))).index:
                examples.append({**identity, "baseline_run": bname, "id": df.loc[i, "id"],
                                 "zh": df.loc[i, "zh"], "ref": df.loc[i, "ref"], "cell": df.loc[i, "cell"],
                                 "hyp_system": df.loc[i, "hyp"], "hyp_baseline": base.loc[i, "hyp"],
                                 "chrf_system": df.loc[i, "chrf"], "chrf_baseline": base.loc[i, "chrf"],
                                 "delta_chrf": delta.loc[i]})
    metric_df = pd.DataFrame(metrics)
    metric_df.to_csv(out / "groups_by_run.csv", index=False)
    numeric = [c for c in ["corpus_bleu", "corpus_chrf", "mean_sentence_chrf", "mean_comet"] if c in metric_df]
    group_keys = ["system_spec", "group_type", "group"]
    means = metric_df.groupby(group_keys, sort=False)[numeric].agg(["mean", "std"]).reset_index()
    means.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in means.columns]
    counts = metric_df.groupby(group_keys, sort=False).agg(n_runs=("run", "size"), n_sentences=("n", "first")).reset_index()
    means = counts.merge(means, on=group_keys, validate="one_to_one")
    means.to_csv(out / "groups_across_seeds.csv", index=False)
    change_df = pd.DataFrame(changes)
    change_df.to_csv(out / "paired_changes_by_run.csv", index=False)
    if len(change_df):
        keys = ["system_spec", "group_type", "group", "metric"]
        agg = change_df.groupby(keys, sort=False).agg(
            n_seed_pairs=("run", "size"), mean_delta=("mean_delta", "mean"),
            seed_delta_sd=("mean_delta", "std"), mean_win_share=("win_share", "mean"),
            mean_loss_share=("loss_share", "mean"), mean_tie_share=("tie_share", "mean")).reset_index()
        agg.to_csv(out / "paired_changes_across_seeds.csv", index=False)
        cells = agg[(agg.group_type == "cell") & (agg.metric == "chrf")]
        for spec in cells.system_spec.unique():
            matrix = np.full((len(LENGTH_LABELS), len(RARITY_LABELS)), np.nan)
            for r in cells[cells.system_spec == spec].itertuples():
                length, rarity = r.group.split("/")
                matrix[LENGTH_LABELS.index(length), RARITY_LABELS.index(rarity)] = r.mean_delta
            fig, ax = plt.subplots(figsize=(5.5, 4.8))
            limit = max(1., np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.
            im = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
            ax.set_xticks(range(2), RARITY_LABELS); ax.set_yticks(range(4), LENGTH_LABELS)
            ax.set_title(f"{spec}\nmean paired sentence chrF change")
            for i in range(4):
                for j in range(2):
                    ax.text(j, i, f"{matrix[i,j]:+.2f}" if np.isfinite(matrix[i,j]) else "no data",
                            ha="center", va="center")
            fig.colorbar(im, ax=ax); fig.tight_layout()
            safe = hashlib.sha256(spec.encode()).hexdigest()[:10]
            fig.savefig(out / f"delta_chrf_{safe}.png", dpi=160); plt.close(fig)
    pd.DataFrame(corrs).to_csv(out / "coverage_score_correlations.csv", index=False)
    pd.DataFrame(examples).to_csv(out / "paired_examples_by_run.csv", index=False)
    write_json(out / "quantitative_manifest.json", {
        "test_set": test_set, "comparison": comparison, "baseline": baseline,
        "systems": {s: [{"run": n, **m} for n, _, m in es] for s, es in runs.items()},
        "bootstrap_draws": n_boot, "seed": seed,
        "scored_file_sha256": {str(p): file_hash(p) for s in specs
                               for p in sorted(Path(scored_dir).glob(f"{s}/{test_set}.csv"))},
        "limitations": ["Intervals resample sentences only, not seeds or documents; no multiple-comparison adjustment.",
                        "Across-seed SD is descriptive (NaN when only one run).",
                        "Correlations and frequency differences do not establish causation.",
                        "Automatic flags are candidate errors; unknown generation-limit status remains unknown.",
                        "Paired examples show exact run text and scores, never averaged scores beside one seed's text."]})
    return means
