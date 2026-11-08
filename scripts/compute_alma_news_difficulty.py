"""Compute sampling features for the merged ALMA + News training pool.

Run after ``scripts/merge_alma_news.py``::

    python scripts/compute_alma_news_difficulty.py

The rarity reference remains ALMA train, matching the original experiment.  ALMA
rows use leave-one-out counts; News rows do not contribute to the reference
counts.  This keeps the original fixed bin definitions and rarity scale usable.
"""

import math
from collections import Counter
from pathlib import Path

import pandas as pd

from compute_difficulty import subword_frequencies
from data_utils import get_tokenizer, read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data/processed"
FEATURE_DIR = REPO_ROOT / "data/features"
ALMA_PATH = PROCESSED_DIR / "alma_train.jsonl"
MERGED_PATH = PROCESSED_DIR / "alma_news_train.jsonl"
OUTPUT_PATH = FEATURE_DIR / "alma_news_train.csv"


def main():
    missing = [path for path in (ALMA_PATH, MERGED_PATH) if not path.is_file()]
    if missing:
        raise SystemExit("Missing required input file(s):\n  " + "\n  ".join(map(str, missing)))

    alma = read_jsonl(ALMA_PATH)
    merged = read_jsonl(MERGED_PATH)
    tokenizer = get_tokenizer()
    counts, total = subword_frequencies(row["zh"] for row in alma)
    vocab_size = len(tokenizer)

    alma_ids = {row["id"] for row in alma}
    merged_ids = [row["id"] for row in merged]
    if len(merged_ids) != len(set(merged_ids)):
        raise SystemExit("Merged training file contains duplicate IDs")
    if not alma_ids.issubset(merged_ids):
        raise SystemExit("Merged training file does not contain every ALMA training ID")

    records = []
    for row in merged:
        pieces = tokenizer.tokenize(row["zh"])
        own = Counter(pieces) if row["id"] in alma_ids else Counter()
        denominator = total - sum(own.values()) + vocab_size
        rarities = [
            -math.log((counts[piece] - own[piece] + 1) / denominator)
            for piece in pieces
        ]
        records.append(
            {
                "id": row["id"],
                "src_len": len(pieces),
                "tgt_len": len(tokenizer.tokenize(row["en"])),
                "rarity_sum": sum(rarities),
                "rarity_mean": sum(rarities) / max(len(pieces), 1),
                "rarity_max": max(rarities, default=0.0),
                "n_unseen": sum(counts[piece] - own[piece] == 0 for piece in pieces),
            }
        )

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUTPUT_PATH, index=False, float_format="%.4f")
    print(f"ALMA rarity reference: {len(alma):,} rows, {total:,} source subwords")
    print(f"Merged features: {len(records):,} rows")
    print(f"saved features: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
