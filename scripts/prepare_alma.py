"""Step 2 of data preparation: clean ALMA-Human-Parallel zh-en and save it in our format.

The cleaning rules below were chosen after looking at the data with audit_alma.py
(see data/audit/audit_report.txt). Each rule is applied in order; a removed row is
attributed to the FIRST rule that removes it, so the counts add up exactly.

Outputs (in data/processed/):
    alma_train.jsonl            - cleaned training pairs
    alma_dev.jsonl              - cleaned development pairs (for checkpoint selection)
    alma_removed.jsonl          - every removed row with the rule that removed it
    alma_cleaning_report.md     - rules, rationale and counts, ready for the project report

Run from the repository root:
    python scripts/prepare_alma.py
"""
from datasets import load_dataset

from data_utils import MAX_LEN, SPECIAL_TOKENS, n_tokens, normalize, write_jsonl

OUT_DIR = "data/processed"
MIN_RATIO, MAX_RATIO = 1 / 3, 3  # allowed range for (en tokens / zh tokens)

# Human-readable description of each rule, used in the Markdown report.
# Order here = order in which the rules are applied.
RULES = {
    "exact_duplicate": (
        "Exact duplicate pair",
        "A pair (zh, en) identical to an earlier one. Duplicates would be sampled twice "
        "as often, which silently distorts a study about sampling. First copy kept."),
    "duplicate_source": (
        "Duplicate source sentence",
        "Same Chinese source as an earlier pair but a different English translation. Both "
        "translations are valid, but we keep one row per source sentence so every sentence "
        "gets exactly its intended sampling weight. First occurrence kept."),
    "length_ratio": (
        f"Length-ratio outlier (en/zh tokens < {MIN_RATIO:.2f} or > {MAX_RATIO})",
        "Translations are normally of comparable length to their source; extreme ratios "
        "were found (by manual inspection) to be mostly misaligned fragments, e.g. "
        "'我们' -> 'In just one week,'. Known false positive: sources that contain a name "
        "twice (Chinese transliteration + English original) can look too long."),
    "too_long": (
        f"Too long (> {MAX_LEN - SPECIAL_TOKENS} subword tokens on either side)",
        f"Training uses max_length={MAX_LEN}. Longer pairs would be truncated, which teaches "
        "the model to stop translating mid-sentence; length-balanced sampling would show "
        "exactly these pairs more often and amplify the damage."),
}


def clean_split(rows, split):
    """Apply all cleaning rules to one split.

    rows  : list of (original_index, zh, en)
    Returns (kept_rows, removed_rows, n_changed_by_normalisation).
    """
    kept, removed = [], []
    seen_pairs, seen_sources = set(), set()
    n_normalised = 0

    for idx, zh_raw, en_raw in rows:
        # --- Normalisation (does not remove anything, only counts changes) ---
        zh, en = normalize(zh_raw), normalize(en_raw)
        n_normalised += (zh != zh_raw) or (en != en_raw)

        row = {"id": f"alma-{split}-{idx:06d}", "source": "alma", "zh": zh, "en": en}
        zh_len, en_len = n_tokens(zh), n_tokens(en)

        # --- Rules, in order. The first rule that matches decides the reason. ---
        if (zh, en) in seen_pairs:
            reason = "exact_duplicate"
        elif zh in seen_sources:
            reason = "duplicate_source"
        elif not (MIN_RATIO <= en_len / max(zh_len, 1) <= MAX_RATIO):
            reason = "length_ratio"
        elif max(zh_len, en_len) + SPECIAL_TOKENS > MAX_LEN:
            reason = "too_long"
        else:
            reason = None

        # Remember this pair/source even if removed for ratio/length, so that a later
        # copy of a removed pair is reported as a duplicate rather than re-checked.
        seen_pairs.add((zh, en))
        seen_sources.add(zh)

        if reason is None:
            kept.append(row)
        else:
            removed.append({**row, "split": split, "reason": reason})

    return kept, removed, n_normalised


def main():
    data = load_dataset("haoranxu/ALMA-Human-Parallel", "zh-en")
    # Our names for the splits ("validation" is our dev set).
    split_names = {"train": "train", "validation": "dev"}

    all_removed, stats = [], {}
    for hf_name, split in split_names.items():
        rows = [(i, r["translation"]["zh"], r["translation"]["en"])
                for i, r in enumerate(data[hf_name])]
        kept, removed, n_norm = clean_split(rows, split)
        write_jsonl(kept, f"{OUT_DIR}/alma_{split}.jsonl")
        all_removed += removed
        stats[split] = {"before": len(rows), "after": len(kept), "normalised": n_norm,
                        **{r: sum(x["reason"] == r for x in removed) for r in RULES}}
        print(f"{split}: {len(rows)} -> {len(kept)} pairs "
              f"({len(removed)} removed, {n_norm} changed by normalisation)")

    write_jsonl(all_removed, f"{OUT_DIR}/alma_removed.jsonl")
    write_report(stats, all_removed)
    print(f"Saved to {OUT_DIR}/")


def write_report(stats, removed):
    """Write a Markdown summary of the cleaning, meant to be reused in the project report."""
    splits = list(stats)
    lines = [
        "# ALMA-Human-Parallel (zh-en) cleaning report",
        "",
        "Generated by `scripts/prepare_alma.py`. Source: Hugging Face dataset "
        "`haoranxu/ALMA-Human-Parallel`, config `zh-en`; its `validation` split is our dev set.",
        "",
        "## Normalisation (applied to every sentence, removes nothing)",
        "",
        "- Unicode NFC normalisation (canonical composition) and stripping of outer whitespace.",
        "- NFKC was deliberately *not* used: it would convert Chinese full-width punctuation "
        "(，。（）) and full-width digits to their ASCII forms, altering the text.",
        "",
        "## Removal rules (applied in this order; each row counted under the first rule it hits)",
        "",
    ]
    for i, (key, (title, why)) in enumerate(RULES.items(), 1):
        lines += [f"{i}. **{title}** (`{key}`): {why}", ""]

    lines += [
        "Deliberately **kept**: Chinese sources containing many Latin-script names "
        "(news sentences such as 'Desert Island Discs：Tom Daley 因性取向而自卑'), and one English "
        "target that quotes Chinese brand names on purpose.",
        "",
        "## Counts",
        "",
        "| | " + " | ".join(splits) + " |",
        "|---|" + "---|" * len(splits),
        "| Pairs before cleaning | " + " | ".join(str(stats[s]["before"]) for s in splits) + " |",
        "| Changed by normalisation | " + " | ".join(str(stats[s]["normalised"]) for s in splits) + " |",
    ]
    for key, (title, _) in RULES.items():
        lines.append(f"| Removed: {title.split(' (')[0]} | "
                     + " | ".join(str(stats[s][key]) for s in splits) + " |")
    lines += [
        "| **Pairs after cleaning** | " + " | ".join(f"**{stats[s]['after']}**" for s in splits) + " |",
        "",
        "## Removed examples (non-duplicate rules)",
        "",
        "Full list with reasons: `data/processed/alma_removed.jsonl`.",
        "",
    ]
    for r in removed:
        if r["reason"] in ("length_ratio", "too_long"):
            lines.append(f"- `{r['id']}` ({r['reason']}): {r['zh'][:60]} → {r['en'][:60]}")

    with open(f"{OUT_DIR}/alma_cleaning_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
