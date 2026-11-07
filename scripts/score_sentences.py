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
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import sacrebleu

from data_utils import MAX_LEN, SPECIAL_TOKENS, assign_bins, get_tokenizer, read_jsonl
from error_analysis_utils import (align_to, coverage_features, file_hash,
                                  infer_run_metadata, load_frequencies, unique_ids, write_json)

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


def read_hypotheses(path, rows, mode="auto"):
    """JSONL provides verifiable IDs; plain text retains the legacy order contract."""
    path = Path(path)
    if mode == "auto":
        mode = "jsonl" if path.suffix == ".jsonl" else "text"
    if mode == "text":
        hyps = read_lines(path)
        if len(hyps) != len(rows):
            raise ValueError(f"{len(hyps)} translations but {len(rows)} source sentences")
        return hyps, "text_order_assumed", None
    raw = pd.DataFrame(read_jsonl(path))
    raw = align_to(raw, [r["id"] for r in rows], "hypotheses")
    if "hyp" not in raw:
        raise ValueError("JSONL hypotheses require id and hyp")
    for field, original in [("zh", "zh"), ("ref", "en")]:
        if field in raw and raw[field].tolist() != [r[original] for r in rows]:
            raise ValueError(f"JSONL {field} differs from source/reference data")
    if not raw["hyp"].map(lambda x: isinstance(x, str)).all():
        raise ValueError("every hypothesis must be a string (empty strings are allowed)")
    # Only report a generation-limit hit when the generator recorded the fact.
    limit = raw["generation_limit_hit"] if "generation_limit_hit" in raw else None
    if limit is not None and not limit.map(lambda x: isinstance(x, bool)).all():
        raise ValueError("generation_limit_hit must contain explicit JSON booleans")
    return raw["hyp"].tolist(), "jsonl_id_checked", limit


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--system", required=True)
    parser.add_argument("--test_set", required=True, help="wmt22, tatoeba or tatoeba_challenge")
    parser.add_argument("--hyp", help="translations file (default outputs/<system>/<test_set>.txt)")
    parser.add_argument("--comet", help="optional file with one COMET score per line")
    parser.add_argument("--out_dir", default="analysis/scored")
    parser.add_argument("--split", help="processed split, e.g. alma_dev; default <test_set>_test")
    parser.add_argument("--hyp_format", choices=["auto", "text", "jsonl"], default="auto")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--lowfreq_threshold", type=int, default=5)
    parser.add_argument("--condition")
    parser.add_argument("--train_size")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_beams", type=int)
    parser.add_argument("--checkpoint", help="optional checkpoint path for provenance")
    parser.add_argument("--overwrite", action="store_true", help="explicitly replace an existing scored file")
    args = parser.parse_args()

    out = Path(args.out_dir) / args.system / f"{args.test_set}.csv"
    if out.exists() and not args.overwrite:
        parser.error(f"{out} exists; choose a new --out_dir or explicitly use --overwrite")
    split = args.split or f"{args.test_set}_test"
    text_path = Path(args.data_dir) / f"{split}.jsonl"
    feature_path = Path(args.features_dir) / f"{split}.csv"
    hyp_path = Path(args.hyp or f"outputs/{args.system}/{args.test_set}.txt")
    rows = read_jsonl(text_path)
    unique_ids(pd.DataFrame(rows), "processed data")
    features = assign_bins(align_to(pd.read_csv(feature_path), [r["id"] for r in rows], "features"))
    hyps, alignment, generation_limit = read_hypotheses(hyp_path, rows, args.hyp_format)

    df = features[["id", "src_len", "tgt_len", "rarity_mean", "length_bin", "rarity_bin", "cell"]].copy()
    df.insert(1, "zh", [r["zh"] for r in rows])
    df.insert(2, "ref", [r["en"] for r in rows])  # WMT22: Annotator A, as in the course guidelines
    df.insert(3, "hyp", hyps)
    metadata = infer_run_metadata(args.system)
    for key in metadata:
        if getattr(args, key) is not None:
            metadata[key] = getattr(args, key)
        df[key] = metadata[key]
    df["system"] = args.system
    df["test_set"] = args.test_set
    for col in ["domain", "docid"]:
        if any(col in r for r in rows):
            df[col] = [r.get(col, "unknown") for r in rows]
    tokenizer = get_tokenizer()
    counts = load_frequencies(Path(args.features_dir) / "subword_freq.tsv")
    coverage = coverage_features(rows, counts, tokenizer, args.lowfreq_threshold, split == "alma_train")
    if df["src_len"].tolist() != coverage["coverage_src_len"].tolist():
        raise ValueError("saved source lengths differ from tokenizer; regenerate features")
    df = df.merge(coverage, on="id", validate="one_to_one")
    df["input_exceeds_max_length"] = df.src_len + SPECIAL_TOKENS > MAX_LEN
    df["hyp_subword_len"] = [len(tokenizer.tokenize(h)) for h in hyps]
    # This is a candidate flag, NOT proof of hitting max_length or of omission.
    df["output_near_training_limit"] = df.hyp_subword_len >= MAX_LEN - SPECIAL_TOKENS
    if generation_limit is not None:
        df["generation_limit_hit"] = generation_limit.to_numpy()

    df["chrf"] = [sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(df["hyp"], df["ref"])]
    df["bleu"] = [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(df["hyp"], df["ref"])]
    if args.comet:
        comet = [float(x) for x in read_lines(args.comet)]
        if len(comet) != len(rows):
            raise SystemExit(f"{len(comet)} COMET scores but {len(rows)} test sentences")
        df["comet"] = comet
        if not pd.Series(comet).map(lambda x: float("-inf") < x < float("inf")).all():
            raise ValueError("COMET scores must be finite")

    df["len_ratio"] = [len(h.split()) / max(len(r.split()), 1) for h, r in zip(df["hyp"], df["ref"])]
    df["empty"] = df["hyp"].str.strip() == ""
    df["has_cjk"] = df["hyp"].apply(lambda h: bool(CJK.search(h)))
    df["missing_nums"] = [missing_numbers(z, h) for z, h in zip(df["zh"], df["hyp"])]
    df["repeated"] = df["hyp"].apply(has_repetition)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.4f")
    write_json(out.with_suffix(".meta.json"), {
        "system": args.system, "split": split, "metadata": metadata,
        "alignment": alignment, "checkpoint": args.checkpoint,
        "lowfreq_threshold": args.lowfreq_threshold,
        "inputs": {str(p): file_hash(p) for p in [text_path, feature_path, hyp_path,
                   Path(args.features_dir) / "subword_freq.tsv"] + ([Path(args.comet)] if args.comet else [])},
        "sacrebleu_version": sacrebleu.__version__,
        "warning": "Automatic flags are screening candidates, not human-confirmed errors. "
                   "ALMA-unseen does not imply unseen in pretraining."})
    print(f"{args.system} on {args.test_set}: {len(df)} sentences, mean chrF {df['chrf'].mean():.2f}, "
          f"mean sentence BLEU {df['bleu'].mean():.2f} -> {out}")


if __name__ == "__main__":
    main()
