"""Clean and deterministically sample News Commentary v16 for zh->en training.

The input is the OPUS ``news-commentary-v16.en-zh.tsv.gz`` file (English in
column 1, Chinese in column 2).  The script applies the same light normalisation and M2M-100 length
measurement as the rest of this repository, removes duplicate/noisy pairs and
protects the ALMA dev and evaluation sets from train/evaluation overlap.

Run from the repository root, after ``scripts/prepare_data.sh``::

    python scripts/prepare_news_commentary.py

Outputs in ``data/processed/`` (ignored by Git):

* ``news_commentary_train.jsonl``: the deterministic 15,000-pair sample;
* ``news_commentary_removed.jsonl``: rejected rows and their first rejection reason;
* ``news_commentary_cleaning_report.md``: provenance, parameters and counts.
"""

import gzip
import hashlib
import re
from collections import Counter
from pathlib import Path

from data_utils import MAX_LEN, SPECIAL_TOKENS, n_tokens, normalize, read_jsonl, write_jsonl


INPUT_PATH = Path(
    "/mnt/scratch/e/e1504054/mt_external_data/raw/"
    "news-commentary-v16.en-zh.tsv.gz"
)
OUTPUT_DIR = Path("data/processed")
PROTECTED_PATHS = [
    Path("data/processed/alma_dev.jsonl"),
    Path("data/processed/wmt22_test.jsonl"),
    Path("data/processed/tatoeba_test.jsonl"),
    Path("data/processed/tatoeba_challenge_test.jsonl"),
]
SAMPLE_SIZE = 15_000
SEED = 0
MIN_RATIO, MAX_RATIO = 1 / 3, 3

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN = re.compile(r"[A-Za-z]")

RULES = {
    "malformed_tsv": "Line does not contain exactly two tab-separated columns",
    "empty_side": "Chinese or English side is empty after normalisation",
    "missing_chinese": "Chinese side contains no CJK character",
    "missing_english": "English side contains no Latin letter",
    "length_ratio": f"English/Chinese token ratio is outside [{MIN_RATIO:.2f}, {MAX_RATIO}]",
    "too_long": f"Either side exceeds {MAX_LEN - SPECIAL_TOKENS} M2M-100 tokens",
    "protected_exact": "Chinese source exactly matches an ALMA dev or evaluation source",
    "protected_loose": "Chinese source matches after removing case, punctuation and spaces",
    "exact_duplicate": "Exact duplicate pair; first qualifying occurrence kept",
    "duplicate_source": "Repeated Chinese source; first qualifying occurrence kept",
}


def loose(text):
    """Near-exact match key used by ``prepare_test_sets.py``."""
    return re.sub(r"[\W_]+", "", text.lower())


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protected(paths):
    missing = [path for path in paths if not path.is_file()]
    if missing:
        joined = "\n  ".join(str(path) for path in missing)
        raise SystemExit(
            "Missing protected split file(s):\n  " + joined
            + "\nRun bash scripts/prepare_data.sh first."
        )

    exact, near = set(), set()
    counts = {}
    for path in paths:
        rows = read_jsonl(path)
        sources = [normalize(row["zh"]) for row in rows]
        exact.update(sources)
        near.update(key for source in sources if (key := loose(source)))
        counts[str(path)] = len(rows)
    return exact, near, counts


def iter_tsv(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2:
                yield line_number, None, None, line.rstrip("\r\n")
            else:
                en, zh = fields
                yield line_number, zh, en, None


def deterministic_sample(rows, size, seed):
    """Select rows by stable SHA-256 ranks, then restore original corpus order."""
    if size <= 0:
        raise SystemExit("--sample-size must be positive")
    if len(rows) < size:
        raise SystemExit(
            f"Only {len(rows):,} pairs remain after cleaning; cannot sample {size:,}."
        )

    def rank(row):
        key = f"{seed}\0{row['id']}".encode("utf-8")
        return hashlib.sha256(key).digest()

    selected = sorted(rows, key=rank)[:size]
    return sorted(selected, key=lambda row: row["line_number"])


def clean(path, protected_exact, protected_near):
    kept, removed = [], []
    counts = Counter()
    seen_pairs, seen_sources = set(), set()

    for line_number, zh_raw, en_raw, malformed in iter_tsv(path):
        row_id = f"news-commentary-{line_number:07d}"
        if malformed is not None:
            removed.append({"id": row_id, "reason": "malformed_tsv", "raw": malformed})
            counts["malformed_tsv"] += 1
            continue

        zh, en = normalize(zh_raw), normalize(en_raw)
        row = {
            "id": row_id,
            "source": "news_commentary_v16",
            "zh": zh,
            "en": en,
            "line_number": line_number,
        }

        if not zh or not en:
            reason = "empty_side"
        elif not CJK.search(zh):
            reason = "missing_chinese"
        elif not LATIN.search(en):
            reason = "missing_english"
        else:
            zh_len, en_len = n_tokens(zh), n_tokens(en)
            ratio = en_len / max(zh_len, 1)
            if not (MIN_RATIO <= ratio <= MAX_RATIO):
                reason = "length_ratio"
            elif max(zh_len, en_len) + SPECIAL_TOKENS > MAX_LEN:
                reason = "too_long"
            elif zh in protected_exact:
                reason = "protected_exact"
            elif loose(zh) in protected_near:
                reason = "protected_loose"
            elif (zh, en) in seen_pairs:
                reason = "exact_duplicate"
            elif zh in seen_sources:
                reason = "duplicate_source"
            else:
                reason = None

        if reason is None:
            kept.append(row)
            seen_pairs.add((zh, en))
            seen_sources.add(zh)
        else:
            removed.append({**row, "reason": reason})
            counts[reason] += 1

    return kept, removed, counts


def write_report(path, input_path, input_sha256, protected_counts, total, candidates,
                 selected, removed_counts, sample_size, seed):
    lines = [
        "# News Commentary v16 cleaning report",
        "",
        "Generated by `scripts/prepare_news_commentary.py`.",
        "",
        "## Provenance and reproducibility",
        "",
        f"- Input: `{input_path}` (OPUS `en-zh.tsv.gz`; English column 1, Chinese column 2)",
        f"- Input SHA-256: `{input_sha256}`",
        f"- Requested sample size: {sample_size:,}",
        f"- Sampling seed: {seed}",
        "- Sampling method: take the smallest SHA-256 ranks of `seed + stable row id`; "
        "output is restored to original corpus order.",
        "- Normalisation: Unicode NFC plus stripping outer whitespace (same as ALMA).",
        "",
        "## Protected splits",
        "",
        "Chinese-source exact and loose matches are excluded from training:",
        "",
    ]
    lines.extend(f"- `{name}`: {count:,} rows" for name, count in protected_counts.items())
    lines += [
        "",
        "Loose matching lowercases text and removes punctuation, underscores and whitespace.",
        "",
        "## Cleaning rules and counts",
        "",
        "Rules are applied in the order below; each rejected row is counted once.",
        "",
        "| Rule | Description | Removed |",
        "|---|---|---:|",
    ]
    for key, description in RULES.items():
        lines.append(f"| `{key}` | {description} | {removed_counts[key]:,} |")
    lines += [
        "",
        "## Final counts",
        "",
        f"- Input lines: {total:,}",
        f"- Clean candidate pairs: {candidates:,}",
        f"- Deterministically selected pairs: **{selected:,}**",
        f"- Clean candidates not selected: {candidates - selected:,}",
        "",
        "The full rejected-row audit is `data/processed/news_commentary_removed.jsonl`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not INPUT_PATH.is_file():
        raise SystemExit(f"Input file not found: {INPUT_PATH}")

    protected_exact, protected_near, protected_counts = load_protected(PROTECTED_PATHS)
    input_sha256 = sha256_file(INPUT_PATH)
    candidates, removed, removed_counts = clean(
        INPUT_PATH, protected_exact, protected_near
    )
    selected = deterministic_sample(candidates, SAMPLE_SIZE, SEED)

    # line_number is retained while sampling, but the shared training schema is only
    # id/source/zh/en; the stable id already traces each row to its original line.
    for row in selected:
        row.pop("line_number")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(selected, OUTPUT_DIR / "news_commentary_train.jsonl")
    write_jsonl(removed, OUTPUT_DIR / "news_commentary_removed.jsonl")
    total = len(candidates) + len(removed)
    write_report(
        OUTPUT_DIR / "news_commentary_cleaning_report.md",
        INPUT_PATH,
        input_sha256,
        protected_counts,
        total,
        len(candidates),
        len(selected),
        removed_counts,
        SAMPLE_SIZE,
        SEED,
    )
    print(f"input: {total:,} lines")
    print(f"clean candidates: {len(candidates):,}")
    print(f"selected: {len(selected):,}")
    print(f"saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
