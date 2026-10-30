"""Error analysis, step 1: score every translation sentence by sentence.

Takes the translations of ONE system on ONE test set and writes one row per test
sentence with its source, reference, hypothesis, difficulty features and bins, sentence-level
metrics, and a few automatic error flags. `analyze_errors.py` then compares systems.

Input convention (so that we do not depend on how the evaluation script stores things):
    outputs/<system>/<test_set>.txt     one translation per line, in the SAME order as
                                        data/processed/<test_set>_test.jsonl
    <system> is the run name, e.g. uniform_n5000_seed0 (the name used in runs/), or
    "pretrained" for the unchanged M2M-100.
    <test_set> is wmt22, tatoeba or tatoeba_challenge.

Output:
    analysis/scored/<system>/<test_set>.csv

Columns: id, zh, ref, hyp, src_len, tgt_len, rarity_mean, length_bin, rarity_bin, cell
(from data/features, via assign_bins), plus
    chrf, bleu   sentence-level sacreBLEU chrF and BLEU (0-100). Sentence BLEU is noisy on short
                 sentences, so chrF is the main per-sentence metric. Corpus-level numbers for the
                 report must still come from the official sacrebleu / comet-score commands.
    comet        only with --comet: per-sentence COMET, one float per line in the same order
    len_ratio    words in hypothesis / words in reference (low: omission, high: addition)
    empty        hypothesis is empty
    has_cjk      hypothesis still contains Chinese characters (untranslated text)
    missing_nums digits that appear in the source but nowhere in the hypothesis
    repeated     some word trigram occurs 3 or more times (degenerate repetition)

Run from the repository root:
    python scripts/score_sentences.py --system uniform_n5000_seed0 --test_set wmt22
    python scripts/score_sentences.py --system pretrained --test_set tatoeba --comet comet_scores.txt
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import sacrebleu

from data_utils import assign_bins, read_jsonl

CJK = re.compile(r"[㐀-鿿]")
DIGITS = re.compile(r"\d+")
# Full-width digits are normalised so that "２０２０" in the source matches "2020" in the output.
FULLWIDTH = str.maketrans("０１２３４５６７８９", "0123456789")


def read_lines(path):
    """Read a one-item-per-line text file. Only the final newline is dropped, so an empty
    last line (an empty translation) is kept."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def missing_numbers(source, hyp):
    """Digit strings that occur in the source but not in the hypothesis."""
    wanted = set(DIGITS.findall(source.translate(FULLWIDTH)))
    return len(wanted - set(DIGITS.findall(hyp)))


def has_repetition(hyp):
    """True if some word trigram occurs at least 3 times."""
    words = hyp.split()
    trigrams = Counter(zip(words, words[1:], words[2:]))
    return any(c >= 3 for c in trigrams.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--system", required=True)
    parser.add_argument("--test_set", required=True, help="wmt22, tatoeba or tatoeba_challenge")
    parser.add_argument("--hyp", help="translations file (default outputs/<system>/<test_set>.txt)")
    parser.add_argument("--comet", help="optional file with one COMET score per line")
    parser.add_argument("--out_dir", default="analysis/scored")
    args = parser.parse_args()

    rows = read_jsonl(f"data/processed/{args.test_set}_test.jsonl")
    features = assign_bins(pd.read_csv(f"data/features/{args.test_set}_test.csv"))
    hyps = read_lines(args.hyp or f"outputs/{args.system}/{args.test_set}.txt")
    if len(hyps) != len(rows):
        raise SystemExit(f"{len(hyps)} translations but {len(rows)} test sentences")
    if features["id"].tolist() != [r["id"] for r in rows]:
        raise SystemExit("data/features and data/processed are not in the same order: "
                         "rerun scripts/prepare_data.sh")

    df = features[["id", "src_len", "tgt_len", "rarity_mean", "length_bin", "rarity_bin", "cell"]].copy()
    df.insert(1, "zh", [r["zh"] for r in rows])
    df.insert(2, "ref", [r["en"] for r in rows])  # WMT22: Annotator A, as in the course guidelines
    df.insert(3, "hyp", hyps)

    df["chrf"] = [sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(df["hyp"], df["ref"])]
    df["bleu"] = [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(df["hyp"], df["ref"])]
    if args.comet:
        comet = [float(x) for x in read_lines(args.comet)]
        if len(comet) != len(rows):
            raise SystemExit(f"{len(comet)} COMET scores but {len(rows)} test sentences")
        df["comet"] = comet

    df["len_ratio"] = [len(h.split()) / max(len(r.split()), 1) for h, r in zip(df["hyp"], df["ref"])]
    df["empty"] = df["hyp"].str.strip() == ""
    df["has_cjk"] = df["hyp"].apply(lambda h: bool(CJK.search(h)))
    df["missing_nums"] = [missing_numbers(z, h) for z, h in zip(df["zh"], df["hyp"])]
    df["repeated"] = df["hyp"].apply(has_repetition)

    out = Path(args.out_dir) / args.system / f"{args.test_set}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"{args.system} on {args.test_set}: {len(df)} sentences, mean chrF {df['chrf'].mean():.2f}, "
          f"mean sentence BLEU {df['bleu'].mean():.2f} -> {out}")


if __name__ == "__main__":
    main()
