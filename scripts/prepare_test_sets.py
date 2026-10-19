"""Step 3 of data preparation: build the test sets in our format.

Test sets are used for FINAL evaluation only (model selection uses the ALMA dev set).
They are NEVER filtered: removing hard or odd test sentences would flatter our results.
The only change is the same light normalisation as for training data (NFC + strip).

OFFICIAL test sets (required by the course guidelines; text taken from the Canvas files,
which must be placed in data/raw/canvas/ - see README):
1. WMT22 general, zh->en (1,875 segments): wmttest2022.zh + wmttest2022.AnnA.en (Annotator A).
   We add metadata from the public release (via sacreBLEU), joined by line number after
   checking that every Chinese line matches: "en_ref_b" (reference B, optional extra
   analysis only - official BLEU uses reference A), "domain" and "docid".
2. Tatoeba (469 pairs, Simplified Chinese): tatoeba.zh + tatoeba.en, a sample of the
   Tatoeba corpus chosen by the course.

SUPPLEMENTARY test set (our addition, never reported as a headline number):
3. Tatoeba Challenge test set, release v2023-09-26, Mandarin Simplified rows (cmn_Hans),
   pinned to a fixed git commit, minus any pair whose source also appears in the official
   Tatoeba set. 469 sentences are too few to tell apart systems that differ by < ~1-2 BLEU;
   this larger set of similar short sentences is used to check whether such differences
   between our sampling conditions are real. Column "in_v2020" flags pairs already in the
   2020 version of the file, the version M2M-100's authors removed from pretraining.

Finally, an overlap check of every test set against ALMA train and dev
(exact match, and a "loose" match ignoring case, punctuation and spaces).

Outputs (in data/processed/):
    wmt22_test.jsonl, tatoeba_test.jsonl     - official
    tatoeba_challenge_test.jsonl             - supplementary
    test_overlap.jsonl                       - every test row that overlaps with ALMA (if any)
    test_sets_report.md                      - sources, sizes, lengths, overlap, for the report
    (raw Tatoeba Challenge downloads are kept in data/raw/tatoeba/ for provenance)

Run from the repository root (after prepare_alma.py):
    python scripts/prepare_test_sets.py
"""

import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from sacrebleu.dataset import DATASETS

from data_utils import (
    MAX_LEN,
    SPECIAL_TOKENS,
    n_tokens,
    normalize,
    read_jsonl,
    write_jsonl,
)

OUT_DIR = "data/processed"
CANVAS_DIR = Path("data/raw/canvas")
CANVAS_FILES = ["wmttest2022.zh", "wmttest2022.AnnA.en", "tatoeba.zh", "tatoeba.en"]

# Supplementary Tatoeba Challenge test set (pinned versions).
TATOEBA_COMMIT = "6faf2dab0b7b01a0e08a114dbacbb7deac54988d"  # "updated dev and test data (v2023-09-26)"
TATOEBA_URL = (
    "https://raw.githubusercontent.com/Helsinki-NLP/Tatoeba-Challenge/"
    f"{TATOEBA_COMMIT}/data/test/eng-zho/test.txt"
)
TATOEBA_RAW = Path("data/raw/tatoeba/eng-zho_test_v2023-09-26.txt")
# The 2020 version of the same file (unchanged until the v2021-08-07 update). The 2023 file
# contains every 2020 pair plus newer ones.
TATOEBA_2020_COMMIT = "17a1b96"
TATOEBA_2020_URL = TATOEBA_URL.replace(TATOEBA_COMMIT, TATOEBA_2020_COMMIT)
TATOEBA_2020_RAW = Path("data/raw/tatoeba/eng-zho_test_v2020.txt")
TATOEBA_VARIETY = (
    "cmn_Hans"  # Mandarin, Simplified script (same as the official Tatoeba set)
)


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def check_canvas_files():
    """Stop with a helpful message if the official files have not been put in place."""
    missing = [f for f in CANVAS_FILES if not (CANVAS_DIR / f).exists()]
    if missing:
        sys.exit(
            f"Missing official test files in {CANVAS_DIR}/: {', '.join(missing)}\n"
            "Download them from the CS4248 Canvas Project folder (see README)."
        )


def load_wmt22():
    """Official WMT22 text from Canvas + metadata from the public release (sacreBLEU)."""
    zh_lines = read_lines(CANVAS_DIR / "wmttest2022.zh")
    en_lines = read_lines(CANVAS_DIR / "wmttest2022.AnnA.en")

    dataset = DATASETS["wmt22"]
    dataset.process_to_text(
        "zh-en"
    )  # downloads and extracts on first use (cached afterwards)
    # sacreBLEU stores one text file per field (src, ref:A, ...), one segment per line.
    files = dict(zip(dataset.fieldnames("zh-en"), dataset.get_files("zh-en")))
    meta = {
        name: read_lines(files[name]) for name in ["src", "ref:B", "domain", "docid"]
    }

    # The join below is by line number, so make sure both files are in the same order.
    assert len(zh_lines) == len(en_lines) == len(meta["src"])
    assert all(normalize(a) == normalize(b) for a, b in zip(zh_lines, meta["src"])), (
        "Canvas WMT22 sources do not match the public release line by line"
    )

    return [
        {
            "id": f"wmt22-test-{i:06d}",
            "source": "wmt22",
            "zh": normalize(zh),
            "en": normalize(en),
            "en_ref_b": normalize(ref_b),
            "domain": domain,
            "docid": docid,
        }
        for i, (zh, en, ref_b, domain, docid) in enumerate(
            zip(zh_lines, en_lines, meta["ref:B"], meta["domain"], meta["docid"])
        )
    ]


def load_tatoeba_official():
    """The course's Tatoeba test set: two line-aligned files from Canvas."""
    zh_lines = read_lines(CANVAS_DIR / "tatoeba.zh")
    en_lines = read_lines(CANVAS_DIR / "tatoeba.en")
    assert len(zh_lines) == len(en_lines)
    return [
        {
            "id": f"tatoeba-test-{i:06d}",
            "source": "tatoeba",
            "zh": normalize(zh),
            "en": normalize(en),
        }
        for i, (zh, en) in enumerate(zip(zh_lines, en_lines))
    ]


def read_tatoeba_challenge_file(url, path):
    """Download a Tatoeba Challenge test file once, then yield (line_no, variety, en, zh).
    Each line looks like: <eng>\\t<chinese variety>\\t<English sentence>\\t<Chinese sentence>"""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    for i, line in enumerate(read_lines(path)):
        _, variety, en, zh = line.split("\t")
        yield i, variety, en, zh


def load_tatoeba_challenge(official):
    """Supplementary set: Simplified Mandarin rows of the pinned 2023 Challenge test file,
    excluding sources that also appear in the official Tatoeba set."""
    pairs_2020 = {
        (normalize(zh), normalize(en))
        for _, variety, en, zh in read_tatoeba_challenge_file(
            TATOEBA_2020_URL, TATOEBA_2020_RAW
        )
        if variety == TATOEBA_VARIETY
    }
    official_keys = {loose(r["zh"]) for r in official}

    rows, n_excluded = [], 0
    for i, variety, en, zh in read_tatoeba_challenge_file(TATOEBA_URL, TATOEBA_RAW):
        if variety != TATOEBA_VARIETY:
            continue
        zh, en = normalize(zh), normalize(en)
        if loose(zh) in official_keys:
            n_excluded += 1
            continue
        rows.append(
            {
                "id": f"tatoeba_challenge-test-{i:06d}",
                "source": "tatoeba_challenge",
                "zh": zh,
                "en": en,
                "in_v2020": (zh, en) in pairs_2020,
            }
        )
    return rows, n_excluded


def loose(text):
    """Key for 'near-exact' matching: lowercase, drop punctuation and whitespace.
    (In Python, \\W does not match Chinese characters, so those are kept.)"""
    return re.sub(r"[\W_]+", "", text.lower())


def length_summary(rows):
    """Median / p95 / max source length in subword tokens, plus the number of
    sources too long for MAX_LEN (they would be truncated at translation time)."""
    lens = np.array([n_tokens(r["zh"]) for r in rows])
    too_long = int((lens + SPECIAL_TOKENS > MAX_LEN).sum())
    return f"{np.median(lens):.0f} | {np.percentile(lens, 95):.0f} | {lens.max()} | {too_long}"


def main():
    check_canvas_files()
    test_sets = {"wmt22": load_wmt22(), "tatoeba": load_tatoeba_official()}
    test_sets["tatoeba_challenge"], n_excluded = load_tatoeba_challenge(
        test_sets["tatoeba"]
    )
    for name, rows in test_sets.items():
        write_jsonl(rows, f"{OUT_DIR}/{name}_test.jsonl")
        print(f"{name}: {len(rows)} pairs")

    # --- Overlap with ALMA (train + dev) ------------------------------------------
    alma = {
        split: read_jsonl(f"{OUT_DIR}/alma_{split}.jsonl") for split in ["train", "dev"]
    }
    exact = {split: {r["zh"]: r["id"] for r in rows} for split, rows in alma.items()}
    near = {
        split: {loose(r["zh"]): r["id"] for r in rows} for split, rows in alma.items()
    }

    overlaps, overlap_counts = [], {}
    for name, rows in test_sets.items():
        for split in alma:
            n_exact = n_loose = 0
            for r in rows:
                if r["zh"] in exact[split]:
                    n_exact += 1
                    overlaps.append(
                        {
                            "test_id": r["id"],
                            "alma_id": exact[split][r["zh"]],
                            "match": "exact",
                            "zh": r["zh"],
                        }
                    )
                elif loose(r["zh"]) and loose(r["zh"]) in near[split]:
                    n_loose += 1
                    overlaps.append(
                        {
                            "test_id": r["id"],
                            "alma_id": near[split][loose(r["zh"])],
                            "match": "loose",
                            "zh": r["zh"],
                        }
                    )
            overlap_counts[(name, split)] = (n_exact, n_loose)
            print(
                f"overlap {name} vs ALMA {split}: {n_exact} exact, {n_loose} loose-only"
            )
    write_jsonl(overlaps, f"{OUT_DIR}/test_overlap.jsonl")

    # --- Report ----------------------------------------------------------------
    wmt, tat, chal = (
        test_sets["wmt22"],
        test_sets["tatoeba"],
        test_sets["tatoeba_challenge"],
    )
    lines = [
        "# Test sets",
        "",
        "Generated by `scripts/prepare_test_sets.py`. Test sets are used only for final "
        "evaluation and are **never filtered**; only NFC normalisation and whitespace "
        "stripping are applied (same as training data).",
        "",
        "## Official test sets (course guidelines; text from the CS4248 Canvas Project folder)",
        "",
        f"- **WMT22 general, zh→en** (Kocmi et al., 2022): `wmttest2022.zh` + "
        f"`wmttest2022.AnnA.en`, {len(wmt)} segments, all originally written in Chinese. "
        "Reference = Annotator A (`en`), as required for official BLEU/COMET. Verified "
        "line-by-line against the public release `wmt22-news-systems` v1.1 (via sacreBLEU): "
        "all sources identical; 4 references differ only by a double space that sacreBLEU's "
        "extraction collapses (no effect on BLEU with the 13a tokenizer). Metadata added from "
        "the public release: `en_ref_b` (Annotator B; optional extra analysis only), `domain` "
        "("
        + ", ".join(f"{d} {c}" for d, c in Counter(r["domain"] for r in wmt).items())
        + ") and `docid`.",
        f"- **Tatoeba** (Tiedemann, 2020; Tatoeba corpus, CC-BY 2.0 FR): `tatoeba.zh` + "
        f"`tatoeba.en`, {len(tat)} pairs, Simplified Chinese. A sample chosen by the course: "
        "only a small fraction of it appears in the Tatoeba Challenge test set, so it is not a "
        "subset of that release.",
        "",
        "## Supplementary test set (our addition; never reported as a headline result)",
        "",
        f"- **Tatoeba Challenge test set**, release v2023-09-26, `data/test/eng-zho/test.txt` at "
        f"commit `{TATOEBA_COMMIT[:7]}` of `Helsinki-NLP/Tatoeba-Challenge`, Simplified "
        f"Mandarin rows (`{TATOEBA_VARIETY}`) only, used zh→en: {len(chal)} pairs after "
        f"excluding {n_excluded} whose source also appears in the official Tatoeba set.",
        "  Purpose: the official Tatoeba set (469 pairs) is too small to distinguish systems "
        "that differ by about 1–2 BLEU; this larger set of similar short sentences is used to "
        "check whether differences between our sampling conditions on short sentences are real.",
        f"  `in_v2020` flags the {sum(r['in_v2020'] for r in chal)} pairs already present in the "
        f"2020 version of the file (commit `{TATOEBA_2020_COMMIT}`). M2M-100's authors removed "
        "the Tatoeba Challenge test sentences of that time from pretraining; newer pairs were "
        "not removed and may have been seen in pretraining.",
        "",
        "## Source lengths (M2M-100 subword tokens)",
        "",
        f"| Set | Pairs | Median | p95 | Max | Longer than max_length={MAX_LEN} |",
        "|---|---|---|---|---|---|",
        f"| WMT22 (official) | {len(wmt)} | {length_summary(wmt)} |",
        f"| Tatoeba (official) | {len(tat)} | {length_summary(tat)} |",
        f"| Tatoeba Challenge (supplementary) | {len(chal)} | {length_summary(chal)} |",
        f"| ALMA train (for comparison) | {len(alma['train'])} | {length_summary(alma['train'])} |",
        "",
        "## Repeated source sentences inside a test set (kept, not removed)",
        "",
    ]
    for name, rows in test_sets.items():
        extra = sum(c - 1 for c in Counter(r["zh"] for r in rows).values())
        lines.append(f"- {name}: {extra} extra copies")
    lines += [
        "",
        "## Overlap between test sources and ALMA",
        "",
        "Exact = identical after normalisation; loose = identical after also lowercasing and "
        "removing punctuation and whitespace. Matching rows: `data/processed/test_overlap.jsonl`.",
        "",
        "| Test set | vs ALMA train (exact / loose) | vs ALMA dev (exact / loose) |",
        "|---|---|---|",
    ]
    for name in test_sets:
        tr, dv = overlap_counts[(name, "train")], overlap_counts[(name, "dev")]
        lines.append(f"| {name} | {tr[0]} / {tr[1]} | {dv[0]} / {dv[1]} |")
    with open(f"{OUT_DIR}/test_sets_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
