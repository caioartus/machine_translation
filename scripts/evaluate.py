"""Export zh->en translations and BLEU/chrF scores. Run from the repo root.

Uses the existing train.translate function for matching generation settings.
Defaults to development data. See EVALUATION.md for Slurm and COMET usage.

--save_nbest also writes nbest.jsonl.gz (gzipped to keep it small enough for git; pandas reads
it directly with pd.read_json(path, lines=True)): for every sentence, all `num_beams` final beam candidates
(best first), each with its model log-probability. Greedy runs store their single output the same
way, so greedy and beam outputs of one checkpoint can be compared on the same scale.
"""
import argparse
import contextlib
import gzip
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def load_examples(path, limit=None):
    """Validate all rows before selecting a prefix; preserve original ordering."""
    rows, seen = [], set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Empty JSONL line {line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number}: expected an object")
            for key in ("id", "zh", "en"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Line {line_number}: missing/nonempty string required: {key}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate id: {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise ValueError("Dataset is empty")
    return rows[:limit], len(rows)


def prediction_records(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError("Prediction count does not match source count")
    if any(not isinstance(p, str) for p in predictions):
        raise ValueError("Predictions must be strings")
    return [{"id": row["id"], "zh": row["zh"], "en": row["en"], "prediction": p}
            for row, p in zip(rows, predictions)]


def nbest(model, tokenizer, sources, device, autocast, batch_size, num_beams, max_length):
    """All final beam candidates per source, best first, with the same generation settings as
    train.translate. Per candidate:
      text      decoded translation
      logprob   sum of log p(token) over the generated tokens (incl. </s>; the forced
                language tag counts 0); comparable between greedy and beam outputs
      n_tokens  number of those tokens
      score     what beam search ranks by: logprob / n_tokens ** length_penalty (HF); for
                greedy (no ranking) simply logprob / n_tokens
    """
    import torch

    model.eval()
    pad = tokenizer.pad_token_id
    results = []
    for i in range(0, len(sources), batch_size):
        enc = tokenizer(sources[i : i + batch_size], max_length=max_length, truncation=True,
                        padding=True, return_tensors="pt").to(device)
        with torch.no_grad(), autocast():
            out = model.generate(
                **enc, forced_bos_token_id=tokenizer.get_lang_id("en"), num_beams=num_beams,
                max_length=max_length, num_return_sequences=num_beams,
                output_scores=True, return_dict_in_generate=True,
            )
            # Beam search already stores log-softmax scores; greedy stores logits.
            token_lp = model.compute_transition_scores(
                out.sequences, out.scores, getattr(out, "beam_indices", None),
                normalize_logits=num_beams == 1,
            ).float()
        generated = out.sequences[:, 1:]                 # position 0 is the decoder start token
        mask = generated[:, -token_lp.shape[1]:] != pad  # padding after </s> does not count
        logprob = (token_lp * mask).sum(1)
        n_tokens = mask.sum(1)
        if num_beams > 1:
            score = out.sequences_scores.float()
        else:
            score = logprob / n_tokens
        texts = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
        for j in range(0, len(texts), num_beams):
            results.append([
                {"text": texts[k], "logprob": round(logprob[k].item(), 4),
                 "n_tokens": int(n_tokens[k]), "score": round(score[k].item(), 4)}
                for k in range(j, j + num_beams)
            ])
    return results


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def versions():
    result = {}
    for name in ("torch", "transformers", "tokenizers", "sentencepiece", "sacrebleu"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/m2m100_418M")
    parser.add_argument("--data", type=Path, default=Path("data/processed/alma_dev.jsonl"))
    parser.add_argument("--out_dir", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--limit", type=positive_int, help="First N rows for debugging only")
    parser.add_argument("--batch_size", type=positive_int, default=16)
    parser.add_argument("--num_beams", type=positive_int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--save_nbest", action="store_true",
                        help="Also write nbest.jsonl.gz: all beam candidates with log-probabilities")
    parser.add_argument("--nbest_batch_size", type=positive_int, default=16,
                        help="Keep equal to --batch_size so padding, and so outputs, are identical")
    args = parser.parse_args()
    rows, total_rows = load_examples(args.data, args.limit)
    if args.out_dir.exists():
        parser.error("Output directory already exists. Choose a new --out_dir.")

    import torch
    from sacrebleu.metrics import BLEU, CHRF
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, set_seed
    from data_utils import MAX_LEN
    from train import translate

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available. Submit the GPU Slurm job.")
    set_seed(0)
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if bf16 else contextlib.nullcontext
    tokenizer = M2M100Tokenizer.from_pretrained(args.model)
    tokenizer.src_lang, tokenizer.tgt_lang = "zh", "en"
    model = M2M100ForConditionalGeneration.from_pretrained(args.model).to(device)
    # Fix the beam-search settings explicitly instead of inheriting them from the checkpoint's
    # generation_config (M2M-100 ships early_stopping=True). Same values as compare_decoding.py.
    model.generation_config.early_stopping = False
    model.generation_config.length_penalty = 1.0
    model.generation_config.do_sample = False

    # Report input truncation without filtering or reordering any test examples.
    too_long = sum(len(tokenizer(r["zh"], truncation=False)["input_ids"]) > MAX_LEN for r in rows)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "data": str(args.data),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "n_total": total_rows, "n_evaluated": len(rows), "limit": args.limit,
        "partial_dataset": len(rows) < total_rows, "reference_field": "en",
        "batch_size": args.batch_size, "num_beams": args.num_beams,
        "early_stopping": model.generation_config.early_stopping,
        "length_penalty": model.generation_config.length_penalty,
        "max_length": MAX_LEN, "source_truncation_count": too_long,
        "device": device, "autocast": "bfloat16" if bf16 else "disabled",
        "packages": versions(),
    }
    write_json(args.out_dir / "metadata.json", metadata)
    print(f"Evaluating {len(rows)}/{total_rows} examples on {device}; truncated sources: {too_long}", flush=True)
    predictions = translate(model, tokenizer, [r["zh"] for r in rows], device, autocast,
                            batch_size=args.batch_size, num_beams=args.num_beams)
    records = prediction_records(rows, predictions)
    with (args.out_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    references = [[r["en"] for r in rows]]
    bleu, chrf = BLEU(tokenize="13a"), CHRF(word_order=0)
    bleu_score = bleu.corpus_score(predictions, references)
    chrf_score = chrf.corpus_score(predictions, references)
    metrics = {
        "n_evaluated": len(rows), "partial_dataset": len(rows) < total_rows,
        "bleu": {"score": bleu_score.score, "signature": str(bleu.get_signature())},
        "chrf": {"score": chrf_score.score, "signature": str(chrf.get_signature())},
    }
    write_json(args.out_dir / "metrics.json", metrics)

    if args.save_nbest:
        # Separate pass, so predictions.jsonl and the metrics above come from the unchanged
        # train.translate. The best candidate should equal the prediction; the share that does
        # is recorded in metadata.json (bf16 batching can very rarely break a tie differently).
        print("Saving n-best lists", flush=True)
        candidates = nbest(model, tokenizer, [r["zh"] for r in rows], device, autocast,
                           args.nbest_batch_size, args.num_beams, MAX_LEN)
        with gzip.open(args.out_dir / "nbest.jsonl.gz", "wt", encoding="utf-8") as stream:
            for row, pred, cands in zip(rows, predictions, candidates):
                stream.write(json.dumps({"id": row["id"], "prediction": pred, "candidates": cands},
                                        ensure_ascii=False) + "\n")
        same = sum(c[0]["text"] == p for c, p in zip(candidates, predictions))
        metadata["nbest_saved"] = True
        metadata["nbest_top1_matches_prediction"] = same / len(rows)
        print(f"n-best top candidate equals the prediction for {same}/{len(rows)} sentences", flush=True)
    metadata["status"] = "complete"
    write_json(args.out_dir / "metadata.json", metadata)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Finished. Results: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
