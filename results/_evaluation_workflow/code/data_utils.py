"""Shared helpers for all data-preparation scripts.

Everything that must be done *identically* for every dataset (training data, dev,
test sets, later UM-Corpus/WMT) lives here, so that e.g. ALMA and WMT22 are
normalised and tokenised in exactly the same way.

The common file format is JSON Lines (one JSON object per line):
    {"id": "alma-train-000042", "source": "alma", "zh": "...", "en": "..."}
  - id     : unique and stable. The number is the row's index in the ORIGINAL
             dataset (before cleaning), so any row can be traced back.
  - source : which corpus the pair comes from (lets us filter or join sources).
Test sets may add extra columns (e.g. "domain" for WMT22).
"""
import json
import unicodedata
from functools import lru_cache
from pathlib import Path

MODEL_NAME = "facebook/m2m100_418M"

# Maximum sequence length used in training, INCLUDING the 2 special tokens that
# M2M-100 adds (a language tag at the start and </s> at the end).
MAX_LEN = 256
SPECIAL_TOKENS = 2


@lru_cache(maxsize=1)
def get_tokenizer():
    """Load the M2M-100 tokenizer once and reuse it (loading is slow)."""
    from transformers import M2M100Tokenizer
    return M2M100Tokenizer.from_pretrained(MODEL_NAME)


def n_tokens(text):
    """Number of subword tokens M2M-100 splits `text` into (without special tokens).

    This is the length the model actually 'sees', and it works for Chinese
    without any word segmentation.
    """
    return len(get_tokenizer().tokenize(text))


def normalize(text):
    """Light, safe text normalisation applied to every sentence of every dataset.

    1. Unicode NFC: makes characters that look identical also identical in bytes
       (e.g. 'é' stored as one character vs 'e' + a combining accent). This makes
       duplicate and overlap checks reliable. It does NOT change Chinese
       full-width punctuation (，。“”), unlike NFKC, which we deliberately avoid.
    2. Strip leading/trailing whitespace.
    """
    return unicodedata.normalize("NFC", text).strip()


def write_jsonl(rows, path):
    """Write a list of dicts to a .jsonl file (UTF-8, Chinese kept readable)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    """Read a .jsonl file back into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --- Difficulty groups ("bins") ------------------------------------------------
# Used by BOTH the training sampler and the evaluation breakdowns, so that e.g.
# "short" means exactly the same thing everywhere. The edges are fixed numbers
# chosen once on the full ALMA training pool (see make_bins.py for the rationale);
# they do not change with the training-set size.
#
# Length = number of source subword tokens (feature "src_len"), upper bounds inclusive:
#   short <= 15 < mid_short <= 30 < mid_long <= 60 < long
LENGTH_EDGES = [15, 30, 60]
LENGTH_LABELS = ["short", "mid_short", "mid_long", "long"]
# Rarity = average rarity per source subword (feature "rarity_mean", in nats):
#   typical <= 7.9 < rare      (7.9 is about the 90th percentile of ALMA train)
RARITY_EDGES = [7.9]
RARITY_LABELS = ["typical", "rare"]


def assign_bins(features):
    """Add length_bin, rarity_bin and cell columns to a features DataFrame.

    `features` is a DataFrame read from data/features/<split>.csv. A "cell" is one
    combination of length bin and rarity bin, e.g. "short/rare" (4 x 2 = 8 cells).
    Returns a new DataFrame; the input is not modified.
    """
    import pandas as pd  # imported here so the rest of this module works without pandas

    out = features.copy()
    inf = float("inf")
    # pd.cut with right=True: a value equal to an edge goes into the LOWER bin (<= edge).
    out["length_bin"] = pd.cut(out["src_len"], [-inf, *LENGTH_EDGES, inf],
                               labels=LENGTH_LABELS, right=True).astype(str)
    out["rarity_bin"] = pd.cut(out["rarity_mean"], [-inf, *RARITY_EDGES, inf],
                               labels=RARITY_LABELS, right=True).astype(str)
    out["cell"] = out["length_bin"] + "/" + out["rarity_bin"]
    return out


def all_cells():
    """The 8 cell names in a fixed order (length-major), e.g. for tables."""
    return [f"{l}/{r}" for l in LENGTH_LABELS for r in RARITY_LABELS]
