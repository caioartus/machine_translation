"""Loss-based difficulty: score every training pair with the PRETRAINED M2M-100.

For each (zh, en) pair we compute the teacher-forced cross-entropy of the English reference
given the Chinese source, using the unchanged pretrained model (eval mode, no dropout).
A high loss means the base model finds the pair hard (or the pair is noisy/misaligned).

Tokenisation and truncation are exactly those used in training (make_collate from train.py).

Output, data/features/<stem>_loss.csv, one row per id:
    id         same ids as data/processed/<stem>.jsonl
    n_tgt      number of target tokens scored (after truncation, incl. </s>)
    loss_sum   summed negative log-likelihood over the target tokens (nats)
    loss_mean  loss_sum / n_tgt  (per-token loss, the main difficulty score)

Run from the repository root:
    python scripts/compute_loss.py                       # ALMA train
    python scripts/compute_loss.py --dataset alma_news   # ALMA + News Commentary train
    python scripts/compute_loss.py --limit 200           # quick smoke test
"""
import argparse
import contextlib
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import M2M100ForConditionalGeneration

from data_utils import MODEL_NAME, get_tokenizer, read_jsonl
from train import make_collate

STEMS = {"alma": "alma_train", "alma_news": "alma_news_train", "alma_dev": "alma_dev"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(STEMS), default="alma")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--limit", type=int, default=None, help="score only the first N pairs (testing)")
    p.add_argument("--out", default=None, help="default: data/features/<stem>_loss.csv")
    return p.parse_args()


@torch.no_grad()
def score(model, loader, device, autocast):
    """Per-sentence summed NLL and number of target tokens."""
    sums, counts = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast():
            logits = model(**batch).logits  # (B, T, V); decoder inputs = shifted labels
        labels = batch["labels"]
        nll = F.cross_entropy(
            logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none"
        )  # (B, T), 0 at padding
        sums.extend(nll.sum(dim=1).tolist())
        counts.extend((labels != -100).sum(dim=1).tolist())
    return sums, counts


def main():
    args = parse_args()
    stem = STEMS[args.dataset]
    in_path = Path("data/processed") / f"{stem}.jsonl"
    out_path = Path(args.out or f"data/features/{stem}_loss.csv")
    if not in_path.is_file():
        raise SystemExit(f"Missing input: {in_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_bf16_supported():
        autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext
    print(f"device: {device}")

    rows = read_jsonl(in_path)
    if args.limit:
        rows = rows[: args.limit]
    # Sort by length so batches have little padding (much faster); order is restored by id.
    rows = sorted(rows, key=lambda r: len(r["zh"]) + len(r["en"]))
    print(f"{in_path}: {len(rows)} pairs")

    tokenizer = get_tokenizer()
    tokenizer.src_lang, tokenizer.tgt_lang = "zh", "en"
    loader = DataLoader(
        rows, batch_size=args.batch_size, shuffle=False, collate_fn=make_collate(tokenizer)
    )
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device).eval()

    start = time.time()
    sums, counts = score(model, loader, device, autocast)
    print(f"scored in {time.time() - start:.0f}s")

    out = pd.DataFrame(
        {"id": [r["id"] for r in rows], "n_tgt": counts, "loss_sum": sums}
    )
    out["loss_mean"] = out["loss_sum"] / out["n_tgt"]
    out = out.sort_values("id").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(out_path, index=False)

    print(f"wrote {out_path}")
    print(out["loss_mean"].describe(percentiles=[0.1, 0.5, 0.8, 0.9, 0.95, 0.99]).round(3))


if __name__ == "__main__":
    main()
