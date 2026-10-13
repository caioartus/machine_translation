"""Shared helpers for all data-preparation scripts.

Everything that must be done *identically* for every dataset (training data, dev,
test sets, later UM-Corpus/WMT) lives here, so that e.g. ALMA and WMT22 are
normalised and tokenised in exactly the same way.

The common file format is JSON Lines (one JSON object per line):
    {"id": "alma-train-000042", "source": "alma", "zh": "...", "en": "..."}
  - id     : unique and stable. The number is the row's index in the ORIGINAL
             dataset (before cleaning), so any row can be traced back.
  - source : which corpus the pair comes from (lets us filter or join sources).
Test sets may add extra columns (e.g. "script" for Tatoeba).
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
