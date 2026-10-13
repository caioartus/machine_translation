"""Step 1 of data preparation: look at ALMA-Human-Parallel zh-en before changing anything.

This script does NOT modify the data. It measures it and flags possible problems,
so that the cleaning rules in prepare_alma.py are based on evidence.

Outputs (in data/audit/):
    audit_report.txt   - every number printed below, saved for later reference
    lengths.png        - histograms of source/target lengths (train vs dev)
    length_ratio.png   - histogram of target/source length ratio

Run from the repository root:
    python scripts/audit_alma.py
"""

import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed (also works on the cluster)
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import M2M100Tokenizer

OUT_DIR = Path("data/audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "facebook/m2m100_418M"
MAX_LEN = 128  # max_length used in simple_train.py; longer inputs get truncated


# --- Print to the terminal and to the report file at the same time ----------
report = open(OUT_DIR / "audit_report.txt", "w", encoding="utf-8")


def log(*args):
    print(*args)
    print(*args, file=report)


# --- Helpers to recognise scripts -------------------------------------------
# CJK Unified Ideographs = the block containing (almost) all Chinese characters.
CJK = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-z]")
# Control characters (invisible, e.g. \x00) that should never appear in text.
# \t \n \r are excluded here and counted separately as "internal whitespace".
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def cjk_fraction(text):
    """Share of non-space characters that are Chinese characters."""
    chars = [c for c in text if not c.isspace()]
    return sum(bool(CJK.match(c)) for c in chars) / max(len(chars), 1)


def percentiles(values):
    """Compact summary of a list of numbers."""
    p = np.percentile(values, [0, 5, 25, 50, 75, 95, 99, 100])
    return (
        "min {:.0f} | p5 {:.0f} | p25 {:.0f} | median {:.0f} | p75 {:.0f} | "
        "p95 {:.0f} | p99 {:.0f} | max {:.0f} | mean {:.1f}"
    ).format(*p, np.mean(values))


def show_examples(title, rows, k=5):
    """Print up to k flagged pairs so we can judge by eye whether the rule is sensible."""
    log(f"    e.g. ({title}):")
    for zh, en in rows[:k]:
        log(f"      zh: {zh[:120]!r}")
        log(f"      en: {en[:120]!r}")


# --- Load data and tokenizer --------------------------------------------------
data = load_dataset("haoranxu/ALMA-Human-Parallel", "zh-en")
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)

# Pull the pairs out of the nested {"translation": {"zh":..., "en":...}} format.
splits = {
    name: [(r["translation"]["zh"], r["translation"]["en"]) for r in data[name]]
    for name in data
}

lengths = {}  # kept for the plots at the end

for name, pairs in splits.items():
    log("=" * 80)
    log(f"SPLIT: {name}   ({len(pairs)} pairs)")
    log("=" * 80)

    zh = [p[0] for p in pairs]
    en = [p[1] for p in pairs]

    # 1) Lengths. "Subword tokens" = what M2M-100 actually sees (no special tokens).
    #    "Characters" is a tokenizer-independent sanity check.
    zh_tok = [len(tokenizer.tokenize(s)) for s in zh]
    en_tok = [len(tokenizer.tokenize(s)) for s in en]
    lengths[name] = (zh_tok, en_tok)
    log("\n[Lengths]")
    log("  zh subword tokens :", percentiles(zh_tok))
    log("  en subword tokens :", percentiles(en_tok))
    log("  zh characters     :", percentiles([len(s) for s in zh]))
    log("  en words (spaces) :", percentiles([len(s.split()) for s in en]))
    over = [i for i, n in enumerate(zh_tok) if n + 2 > MAX_LEN]  # +2: lang token + </s>
    log(f"  zh longer than max_length={MAX_LEN} (would be truncated): {len(over)}")
    over_en = [i for i, n in enumerate(en_tok) if n + 2 > MAX_LEN]
    log(f"  en longer than max_length={MAX_LEN} (would be truncated): {len(over_en)}")

    # 2) Possible quality problems. Each check counts rows and shows examples.
    log("\n[Possible problems]")

    def check(title, mask):
        flagged = [p for p, m in zip(pairs, mask) if m]
        log(f"  {title}: {len(flagged)}")
        if flagged:
            show_examples(title, flagged)

    check(
        "empty or whitespace-only side",
        [not a.strip() or not b.strip() for a, b in pairs],
    )
    check(
        "leading/trailing whitespace",
        [a != a.strip() or b != b.strip() for a, b in pairs],
    )
    check(
        "internal newline/tab", [bool(re.search(r"[\n\t\r]", a + b)) for a, b in pairs]
    )
    check("control characters", [bool(CONTROL.search(a + b)) for a, b in pairs])
    check("source identical to target", [a.strip() == b.strip() for a, b in pairs])
    check("zh side has NO Chinese characters", [not CJK.search(a) for a, _ in pairs])
    check(
        "zh side mostly non-Chinese (<30% CJK)",
        [0 < cjk_fraction(a) < 0.3 for a, _ in pairs],
    )
    check(
        "en side contains Chinese characters", [bool(CJK.search(b)) for _, b in pairs]
    )
    check("en side has no Latin letters", [not LATIN.search(b) for _, b in pairs])

    # Length ratio: a translation is usually of comparable length to its source.
    # Extreme ratios often mean misaligned pairs (e.g. one side is a fragment).
    ratio = np.array([e / max(z, 1) for z, e in zip(zh_tok, en_tok)])
    log("  en/zh token ratio :", percentiles(ratio * 100).replace("min", "(x100) min"))
    check("ratio < 0.33 (en much shorter)", list(ratio < 1 / 3))
    check("ratio > 3   (en much longer)", list(ratio > 3))

    # 3) Duplicates.
    log("\n[Duplicates]")
    pair_counts = Counter(pairs)
    zh_counts = Counter(zh)
    log(
        f"  exact duplicate pairs (extra copies): {sum(c - 1 for c in pair_counts.values())}"
    )
    log(
        f"  duplicate zh sources  (extra copies): {sum(c - 1 for c in zh_counts.values())}"
    )
    # Same source, different targets: not wrong (several valid translations), but worth knowing.
    multi = [
        s
        for s, c in zh_counts.items()
        if c > 1 and len({e for z, e in pairs if z == s}) > 1
    ]
    log(f"  zh sources with >1 different translation: {len(multi)}")
    for s in multi[:3]:
        log(f"      zh: {s[:80]!r}")
        for z, e in pairs:
            if z == s:
                log(f"        -> {e[:80]!r}")

# --- Overlap between train and dev (dev must be unseen for model selection) ----
log("=" * 80)
train_zh = {z.strip() for z, _ in splits["train"]}
dev_zh = [z.strip() for z, _ in splits["validation"]]
log(
    f"dev sources also in train (exact match): {sum(z in train_zh for z in dev_zh)} / {len(dev_zh)}"
)

# --- Plots ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, side, idx in [(axes[0], "zh (source)", 0), (axes[1], "en (target)", 1)]:
    bins = np.arange(0, 160, 4)
    for name, (color) in [("train", "tab:blue"), ("validation", "tab:orange")]:
        ax.hist(
            lengths[name][idx],
            bins=bins,
            density=True,
            alpha=0.5,
            color=color,
            label=name,
        )
    ax.set_xlabel("M2M-100 subword tokens")
    ax.set_ylabel("share of sentences")
    ax.set_title(f"{side} length")
    ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "lengths.png", dpi=120)

zh_tok, en_tok = lengths["train"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(np.log2(np.array(en_tok) / np.maximum(zh_tok, 1)), bins=60)
ax.set_xlabel("log2(en tokens / zh tokens)   (0 = same length)")
ax.set_ylabel("pairs")
ax.set_title("Train: target/source length ratio")
fig.tight_layout()
fig.savefig(OUT_DIR / "length_ratio.png", dpi=120)

log(f"\nSaved report and plots to {OUT_DIR}/")
report.close()
