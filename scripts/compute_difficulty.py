"""Step 4 of data preparation: precompute difficulty features for every example.

We save RAW measurements (e.g. "34 tokens"), not sampling weights. Weights depend on
choices we want to change freely later (bin edges, cap, training-set size), and the
training code can turn features into weights in a fraction of a second.

Features (one row per example, keyed by "id"):
    src_len      number of M2M-100 subword tokens in the Chinese source
    tgt_len      number of subword tokens in the English target / reference
    rarity_sum   sum over source subwords of -log p(subword)     (Platanios et al., 2019, eq. 3)
    rarity_mean  rarity_sum / src_len: average rarity PER subword (roughly independent of length)
    rarity_max   rarity of the single rarest subword in the source
    n_unseen     number of source subwords never seen in ALMA train

Where p(subword) comes from (Platanios et al., eq. 2, with subwords instead of words):
    - counts are taken from the Chinese side of the cleaned ALMA TRAINING set only, and
      reused for every split, so a sentence's difficulty does not depend on which set it is in;
    - add-one (Laplace) smoothing: p(w) = (count(w) + 1) / (N + V), where N = total number of
      subword tokens in ALMA train and V = size of the M2M-100 vocabulary. This gives subwords
      never seen in training a small non-zero probability instead of p = 0 (-log 0 = infinity).
    - logs are natural logs, so rarities are in "nats".

Outputs (in data/features/):
    alma_train.csv, alma_dev.csv, wmt22_test.csv, tatoeba_test.csv, tatoeba_challenge_test.csv
    subword_freq.tsv   - every subword seen in ALMA train with its count and -log p (inspectable)

Run from the repository root (after prepare_alma.py and prepare_test_sets.py):
    python scripts/compute_difficulty.py
"""
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from data_utils import get_tokenizer, read_jsonl

IN_DIR = Path("data/processed")
OUT_DIR = Path("data/features")
SPLITS = ["alma_train", "alma_dev", "wmt22_test", "tatoeba_test", "tatoeba_challenge_test"]


def subword_frequencies(sentences):
    """Count subword tokens over a list of sentences. Returns (Counter, total count N)."""
    tokenizer = get_tokenizer()
    counts = Counter()
    for s in sentences:
        counts.update(tokenizer.tokenize(s))
    return counts, sum(counts.values())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer()
    data = {split: read_jsonl(IN_DIR / f"{split}.jsonl") for split in SPLITS}

    # --- 1. Subword probabilities from the ALMA training sources ------------------
    counts, N = subword_frequencies(r["zh"] for r in data["alma_train"])
    V = len(tokenizer)  # vocabulary size of M2M-100 (all subwords it can produce)

    def neg_log_p(piece):
        """Rarity of one subword: -log of its smoothed probability. Higher = rarer."""
        return -math.log((counts[piece] + 1) / (N + V))

    print(f"ALMA train: {N} source subword tokens, {len(counts)} distinct subwords "
          f"(vocabulary size V = {V})")
    print(f"rarity of a subword seen once: {neg_log_p('__never_seen__') - math.log(2):.2f} nats, "
          f"never seen: {neg_log_p('__never_seen__'):.2f} nats")

    # Save the frequency table so anyone can check which subwords count as rare.
    freq = pd.DataFrame(
        [(piece, c, neg_log_p(piece)) for piece, c in counts.most_common()],
        columns=["subword", "count", "neg_log_p"])
    freq.to_csv(OUT_DIR / "subword_freq.tsv", sep="\t", index=False)

    # --- 2. Per-example features for every split ------------------------------------
    for split, rows in data.items():
        records = []
        for r in rows:
            pieces = tokenizer.tokenize(r["zh"])
            rarities = [neg_log_p(p) for p in pieces]
            records.append({
                "id": r["id"],
                "src_len": len(pieces),
                "tgt_len": len(tokenizer.tokenize(r["en"])),
                "rarity_sum": sum(rarities),
                "rarity_mean": sum(rarities) / max(len(pieces), 1),
                "rarity_max": max(rarities, default=0.0),
                "n_unseen": sum(counts[p] == 0 for p in pieces),
            })
        df = pd.DataFrame(records)
        df.to_csv(OUT_DIR / f"{split}.csv", index=False, float_format="%.4f")

        # Short summary: medians, and how strongly each rarity measure follows length.
        # (Spearman correlation compares rankings: 1 = same order, 0 = unrelated.)
        corr = df[["src_len", "rarity_sum", "rarity_mean", "rarity_max"]].corr(method="spearman")
        print(f"\n{split} ({len(df)} rows)  medians: src_len {df.src_len.median():.0f}, "
              f"rarity_mean {df.rarity_mean.median():.2f}, rarity_max {df.rarity_max.median():.2f}; "
              f"rows with an unseen subword: {(df.n_unseen > 0).mean():.1%}")
        print(f"  Spearman correlation with src_len: rarity_sum {corr.loc['src_len', 'rarity_sum']:.2f}, "
              f"rarity_mean {corr.loc['src_len', 'rarity_mean']:.2f}, "
              f"rarity_max {corr.loc['src_len', 'rarity_max']:.2f}")

    print(f"\nSaved features to {OUT_DIR}/")


if __name__ == "__main__":
    main()
