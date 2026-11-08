"""Describe corpus shift using fixed ALMA source-subword frequencies; no model inference.

Run from the repository root. Outputs never change the shared training features.
ALMA training coverage uses leave-one-out; test coverage uses the full ALMA pool.
These descriptive differences do not establish a causal domain effect.
"""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils import assign_bins, get_tokenizer, read_jsonl
from error_analysis_utils import (align_to, coverage_features, file_hash, group_rows,
                                  load_frequencies, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="data/processed")
    p.add_argument("--features_dir", default="data/features")
    p.add_argument("--splits", nargs="+", default=["alma_train", "alma_dev", "wmt22_test", "tatoeba_test"])
    p.add_argument("--lowfreq_threshold", type=int, default=5)
    p.add_argument("--out_dir", default="analysis/corpus_profiles")
    a = p.parse_args()
    out = Path(a.out_dir)
    # Refuse to silently replace prior analytical choices/results.
    if out.exists() and any(out.iterdir()):
        p.error("output directory is not empty; choose a fresh --out_dir")
    freq_path = Path(a.features_dir) / "subword_freq.tsv"
    counts = load_frequencies(freq_path)
    tokenizer = get_tokenizer()
    train = read_jsonl(Path(a.data_dir) / "alma_train.jsonl")
    recounted = Counter(piece for r in train for piece in tokenizer.tokenize(r["zh"]))
    if recounted != counts:
        raise ValueError("ALMA frequency table/tokenizer/training data mismatch; regenerate features")
    out.mkdir(parents=True, exist_ok=True)
    profiles, summaries, bins, top = {}, [], [], []
    provenance = {"frequency_table": file_hash(freq_path),
                  "train_text": file_hash(Path(a.data_dir) / "alma_train.jsonl"),
                  "lowfreq_threshold": a.lowfreq_threshold,
                  "tokenizer": "facebook/m2m100_418M",
                  "definitions": {"unseen": "ALMA count == 0",
                                  "lowfreq": "1 <= ALMA count <= threshold; excludes unseen",
                                  "train": "sentence leave-one-out coverage and existing LOO rarity",
                                  "held_out": "fixed full ALMA training counts"}, "inputs": {}}
    for split in a.splits:
        text_path = Path(a.data_dir) / f"{split}.jsonl"
        feat_path = Path(a.features_dir) / f"{split}.csv"
        rows = read_jsonl(text_path)
        ids = [r["id"] for r in rows]
        features = assign_bins(align_to(pd.read_csv(feat_path), ids, split))
        coverage = coverage_features(rows, counts, tokenizer, a.lowfreq_threshold, split == "alma_train")
        if features["src_len"].tolist() != coverage["coverage_src_len"].tolist():
            raise ValueError(f"{split}: tokenization differs from saved difficulty features")
        df = features.merge(coverage, on="id", validate="one_to_one")
        if any("domain" in r for r in rows):
            df["domain"] = [r.get("domain", "unknown") for r in rows]
        profiles[split] = df
        df.to_csv(out / f"{split}.csv", index=False)
        total = int(df.src_len.sum())
        summaries.append({"split": split, "n": len(df), "source_tokens": total,
                          "length_mean": df.src_len.mean(), "length_median": df.src_len.median(),
                          "length_p95": df.src_len.quantile(.95),
                          "long_sentence_share": (df.length_bin == "long").mean(),
                          "rarity_mean": df.rarity_mean.mean(), "rarity_median": df.rarity_mean.median(),
                          "rare_sentence_share": (df.rarity_bin == "rare").mean(),
                          "unseen_token_share": df.n_unseen_alma.sum() / max(total, 1),
                          "lowfreq_token_share": df.n_lowfreq_alma.sum() / max(total, 1),
                          "sentences_with_unseen_share": (df.n_unseen_alma > 0).mean(),
                          "coverage_reference": coverage.coverage_reference.iloc[0]})
        for kind, label, sub in group_rows(df):
            bins.append({"split": split, "group_type": kind, "group": label,
                         "n": len(sub), "sentence_share": len(sub) / len(df)})
        own = Counter(piece for r in rows for piece in tokenizer.tokenize(r["zh"]))
        # Union with ALMA's top pieces allows a normalized side-by-side comparison.
        candidates = set(dict(own.most_common(100))) | set(dict(counts.most_common(100)))
        for piece in sorted(candidates):
            top.append({"split": split, "subword": piece, "count": own[piece],
                        "per_10000_tokens": 10000 * own[piece] / max(total, 1),
                        "alma_count": counts[piece],
                        "alma_per_10000_tokens": 10000 * counts[piece] / sum(counts.values())})
        provenance["inputs"][split] = {"text": file_hash(text_path), "features": file_hash(feat_path)}
    pd.DataFrame(summaries).to_csv(out / "corpus_summary.csv", index=False)
    pd.DataFrame(bins).to_csv(out / "group_distribution.csv", index=False)
    pd.DataFrame(top).to_csv(out / "top_subwords.csv", index=False)
    for column, xlabel, name in [("src_len", "Source subword tokens", "length_distribution.png"),
                                 ("rarity_mean", "Mean ALMA-relative rarity (nats)", "rarity_distribution.png")]:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for split, df in profiles.items():
            values = np.sort(df[column].to_numpy())
            ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=split)
        ax.set(xlabel=xlabel, ylabel="Cumulative share of sentences", ylim=(0, 1.01))
        ax.legend(); fig.tight_layout(); fig.savefig(out / name, dpi=160); plt.close(fig)
    write_json(out / "provenance.json", provenance)
    table = pd.DataFrame(summaries)
    (out / "README.md").write_text(
        "# Corpus profiles\n\n" + table.to_csv(index=False) +
        "\nALMA training coverage is leave-one-out; held-out coverage uses the complete ALMA pool. "
        "Unseen means unseen during this fine-tuning corpus, not unseen during model pretraining. "
        "Low-frequency excludes unseen. Counts include punctuation and are subwords, not linguistic words. "
        "Top-frequency differences are descriptive and do not establish text domains or causality. "
        "Rarity and bins are reused from the fixed training-based features; test sets never redefine them.\n",
        encoding="utf-8")
    print(table.to_string(index=False))
    print(f"Saved profiles and two figures to {out}")


if __name__ == "__main__":
    main()
