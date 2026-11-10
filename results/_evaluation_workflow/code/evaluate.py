"""Export zh->en translations and BLEU/chrF scores. Run from the repo root.

Uses the existing train.translate function for matching generation settings.
Defaults to development data. See EVALUATION.md for Slurm and COMET usage.
"""
import argparse
import contextlib
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
    parser.add_argument("--batch_size", type=positive_int, default=32)
    parser.add_argument("--num_beams", type=positive_int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
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

    # Match the development decoding comparison explicitly.
    model.generation_config.do_sample = False
    model.generation_config.num_return_sequences = 1
    model.generation_config.length_penalty = 1.0
    model.generation_config.early_stopping = False

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
        "do_sample": model.generation_config.do_sample,
        "num_return_sequences": model.generation_config.num_return_sequences,
        "length_penalty": model.generation_config.length_penalty,
        "early_stopping": model.generation_config.early_stopping,
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
    metadata["status"] = "complete"
    write_json(args.out_dir / "metadata.json", metadata)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Finished. Results: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
