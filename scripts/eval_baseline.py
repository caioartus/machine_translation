"""Dev loss and greedy dev BLEU on alma_dev for a model, to compare training runs against.

By default this scores the UNCHANGED pretrained M2M-100 (the "no fine-tuning" baseline), using
exactly the same functions as train.py, so the numbers are directly comparable with
`best_dev_loss` and `dev_bleu_greedy` in each run's run_info.json.

Run from the repository root:
    python scripts/eval_baseline.py                                   # pretrained baseline
    python scripts/eval_baseline.py --model runs/joint_n1000_seed0/best   # a trained checkpoint

Output: <out_dir>/<name>.json (default: baselines/pretrained.json), also printed.
Needs a GPU for reasonable speed (see job.sbatch).
"""
import argparse
import contextlib
import json
from pathlib import Path

import sacrebleu
import torch
from torch.utils.data import DataLoader
from transformers import M2M100ForConditionalGeneration

from data_utils import MODEL_NAME, get_tokenizer, read_jsonl
from train import dev_loss, make_collate, translate   # same code as in training


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_NAME, help="model name or path to a saved checkpoint")
    p.add_argument("--name", default=None, help="output file name (default: derived from --model)")
    p.add_argument("--out_dir", default="baselines")
    p.add_argument("--dev_n", type=int, default=None, help="only use the first N dev pairs (debugging)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_bf16_supported():
        autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext
    print(f"device: {device}")

    tokenizer = get_tokenizer()
    tokenizer.src_lang, tokenizer.tgt_lang = "zh", "en"
    dev = read_jsonl("data/processed/alma_dev.jsonl")
    if args.dev_n:
        dev = dev[:args.dev_n]

    model = M2M100ForConditionalGeneration.from_pretrained(args.model).to(device)

    loader = DataLoader(dev, batch_size=32, shuffle=False, collate_fn=make_collate(tokenizer))
    loss = dev_loss(model, loader, device, autocast)
    hyps = translate(model, tokenizer, [r["zh"] for r in dev], device, autocast)
    bleu = sacrebleu.corpus_bleu(hyps, [[r["en"] for r in dev]]).score

    result = {"model": args.model, "n_dev": len(dev), "dev_loss": loss, "dev_bleu_greedy": bleu}
    print(json.dumps(result, indent=2))

    name = args.name or ("pretrained" if args.model == MODEL_NAME
                         else Path(args.model).parent.name)
    out = Path(args.out_dir) / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
