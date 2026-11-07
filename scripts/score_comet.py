"""Score evaluate.py predictions without re-running translation.

May run in a separate COMET environment to avoid changing training dependencies.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from evaluate import load_examples, positive_int


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file")
    parser.add_argument("--batch_size", type=positive_int, default=8)
    parser.add_argument("--gpus", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose another --output.")
    rows, _ = load_examples(args.predictions)
    for row in rows:
        if not isinstance(row.get("prediction"), str):
            parser.error(f"Missing prediction string for {row['id']}")
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:
        raise SystemExit("COMET is not available. See EVALUATION.md for a separate scoring environment.") from exc

    model_name = "Unbabel/wmt22-comet-da"
    model = load_from_checkpoint(download_model(model_name))
    output = model.predict(
        [{"src": r["zh"], "mt": r["prediction"], "ref": r["en"]} for r in rows],
        batch_size=args.batch_size, gpus=args.gpus,
    )
    if len(output.scores) != len(rows):
        raise ValueError("COMET returned a different number of scores than input rows")
    result = {
        "model": model_name, "package_version": importlib.metadata.version("unbabel-comet"),
        "predictions": str(args.predictions),
        "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "n_evaluated": len(rows), "batch_size": args.batch_size, "gpus": args.gpus,
        "system_score": float(output.system_score),
        "sentence_scores": [{"id": r["id"], "score": float(s)} for r, s in zip(rows, output.scores)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally overwriting another scoring run.
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"COMET: {output.system_score}; saved {args.output}")


if __name__ == "__main__":
    main()
