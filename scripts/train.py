"""Fine-tune M2M-100 (418M) for zh->en on one ALMA training subset, under one sampling condition.

The only thing that differs between conditions is the sampling weights
Model, loss, optimizer and number of steps are identical

What happens, in order:
  1. Load the subset (ids from data/subsets/), its text, and its difficulty groups.
  2. Turn the condition into per-example sampling weights.
  3. Train for exactly `--steps` optimizer steps; batches are drawn WITH replacement.
  4. Every `--eval_every` steps, compute the dev loss (alma_dev). The checkpoint with the
     lowest dev loss is saved. Step 0 = the unchanged pretrained model.
  5. At the end, reload the best checkpoint and compute greedy dev BLEU.

Run from the repository root, for example:
    python scripts/train.py --condition uniform --size 1000 --seed 0
    python scripts/train.py --condition length  --size 1000 --seed 0
    python scripts/train.py --condition joint   --size 1000 --seed 0

Quick smoke test (about a minute):
    python scripts/train.py --condition joint --size 1000 --steps 20 --eval_every 10 --out_dir runs_test

Outputs, in <out_dir>/<condition>_n<size>_seed<seed>/:
    best/            the selected checkpoint (model + tokenizer)
    dev_curve.csv    step, train_loss (mean since last eval), dev_loss
    run_info.json    all settings, best step, best dev loss, dev BLEU
"""

import argparse
import contextlib
import json
import time
from pathlib import Path

import pandas as pd
import sacrebleu
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    M2M100ForConditionalGeneration,
    get_linear_schedule_with_warmup,
    set_seed,
)

from data_utils import MAX_LEN, MODEL_NAME, assign_bins, get_tokenizer, read_jsonl


# --- Settings -------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=["uniform", "length", "joint"], required=True)
    p.add_argument("--size", default="1000", help="1000 | 2000 | 5000 | full")
    p.add_argument("--seed", type=int, default=0)
    # Budget: must be the same for all conditions at a given size (team decision, placeholders here).
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument(
        "--cap", type=float, default=3.0, help="max sampling weight vs uniform"
    )
    p.add_argument("--out_dir", default="runs")
    return p.parse_args()


# --- Sampling weights -----------------------------------------
def balanced_weights(groups, cap=3.0):
    """Every group gets the same total probability, but no example is drawn more than
    `cap` times as often as under uniform sampling.
    Note: after clipping the weights no longer sum to n, and the sampler renormalises them,
    so the effective cap is slightly above `cap` (and clipped groups are slightly under-balanced)."""
    n_groups = groups.nunique()
    group_size = groups.map(groups.value_counts())      # per example: size of its own group
    ratio = (len(groups) / n_groups) / group_size
    return ratio.clip(upper=cap)


def make_weights(train, condition, cap):
    if condition == "uniform":
        return pd.Series(1.0, index=train.index)
    if condition == "length":
        return balanced_weights(train["length_bin"], cap)
    if condition == "joint":
        return balanced_weights(train["cell"], cap)
    raise ValueError(condition)


# --- Data -----------------------------------------------------------------------
def load_train(size):
    """Training subset as a DataFrame with columns id, zh, en, ..., length_bin, rarity_bin, cell."""
    ids = (
        open(f"data/subsets/seed{{seed}}_n{size}.txt".format(seed=ARGS.seed))
        .read()
        .split()
    )
    text = pd.DataFrame(read_jsonl("data/processed/alma_train.jsonl"))
    feats = assign_bins(pd.read_csv("data/features/alma_train.csv"))
    return text.merge(feats, on="id").set_index("id").loc[ids].reset_index()


def make_collate(tokenizer):
    """Turns a list of {"zh":..., "en":...} into padded tensors.
    `labels` are the English token ids; padding positions are set to -100 so the loss ignores them.
    (The model builds the decoder inputs itself by shifting the labels one step to the right.)"""

    def collate(batch):
        enc = tokenizer(
            [b["zh"] for b in batch],
            text_target=[b["en"] for b in batch],
            max_length=MAX_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc["labels"][enc["labels"] == tokenizer.pad_token_id] = -100
        return enc

    return collate


# --- Evaluation -----------------------------------------------------------------
@torch.no_grad()
def dev_loss(model, loader, device, autocast):
    """Mean cross-entropy per target token over the whole dev set."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast():
            loss = model(**batch).loss  # mean over the non-ignored target tokens
        n = (batch["labels"] != -100).sum().item()
        total_loss += loss.item() * n
        total_tokens += n
    model.train()
    return total_loss / total_tokens


@torch.no_grad()
def translate(model, tokenizer, sources, device, autocast, batch_size=32, num_beams=1):
    model.eval()
    outputs = []
    for i in range(0, len(sources), batch_size):
        enc = tokenizer(
            sources[i : i + batch_size],
            max_length=MAX_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with autocast():
            generated = model.generate(
                **enc,
                forced_bos_token_id=tokenizer.get_lang_id("en"),
                num_beams=num_beams,
                max_length=MAX_LEN,
            )
        outputs += tokenizer.batch_decode(generated, skip_special_tokens=True)
    return outputs


# --- Main -----------------------------------------------------------------------
def main():
    args = ARGS
    set_seed(args.seed)  # python, numpy and torch (incl. dropout, init)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and torch.cuda.is_bf16_supported():
        autocast = lambda: torch.autocast(
            "cuda", dtype=torch.bfloat16
        )  # faster, less memory
    else:
        autocast = contextlib.nullcontext
    print(f"device: {device}")

    run_dir = Path(args.out_dir) / f"{args.condition}_n{args.size}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer: Chinese in, English out.
    tokenizer = get_tokenizer()
    tokenizer.src_lang, tokenizer.tgt_lang = "zh", "en"
    collate = make_collate(tokenizer)

    # Data and sampler.
    train = load_train(args.size)
    dev = read_jsonl("data/processed/alma_dev.jsonl")
    weights = make_weights(train, args.condition, args.cap)
    print(
        f"train: {len(train)} examples, condition '{args.condition}', "
        f"weights min {weights.min():.2f} / max {weights.max():.2f}"
    )

    sampler = WeightedRandomSampler(
        torch.tensor(weights.values, dtype=torch.double),
        num_samples=args.steps * args.batch_size,  # exactly `steps` batches
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train.to_dict("records"),
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collate,
    )
    dev_loader = DataLoader(dev, batch_size=32, shuffle=False, collate_fn=collate)

    # Model, optimizer, learning-rate schedule (linear warm-up, then linear decay to 0).
    # Dropout is already part of the M2M-100 config (0.1); model.train() turns it on.
    model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, args.warmup_steps, args.steps
    )

    # Step 0 = pretrained model, untouched (useful as a baseline).
    best_loss = dev_loss(model, dev_loader, device, autocast)
    curve = [{"step": 0, "train_loss": None, "dev_loss": best_loss}]
    best_step = 0
    model.save_pretrained(run_dir / "best")
    tokenizer.save_pretrained(run_dir / "best")
    print(f"step 0: dev loss {best_loss:.4f} (pretrained)")

    # Training loop.
    running, n_running, start = 0.0, 0, time.time()
    for step, batch in enumerate(train_loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast():
            loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )  # avoids rare huge updates
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        running += loss.item()
        n_running += 1

        if step % args.eval_every == 0 or step == args.steps:
            d_loss = dev_loss(model, dev_loader, device, autocast)
            curve.append(
                {"step": step, "train_loss": running / n_running, "dev_loss": d_loss}
            )
            print(
                f"step {step}/{args.steps}: train loss {running / n_running:.4f}, "
                f"dev loss {d_loss:.4f}, {time.time() - start:.0f}s"
            )
            running, n_running = 0.0, 0
            if d_loss < best_loss:  # same selection rule for every condition
                best_loss, best_step = d_loss, step
                model.save_pretrained(run_dir / "best")
            pd.DataFrame(curve).to_csv(run_dir / "dev_curve.csv", index=False)

    # Final: reload the selected checkpoint and score it on dev (greedy decoding).
    del model, optimizer
    best = M2M100ForConditionalGeneration.from_pretrained(run_dir / "best").to(device)
    hyps = translate(best, tokenizer, [r["zh"] for r in dev], device, autocast)
    bleu = sacrebleu.corpus_bleu(hyps, [[r["en"] for r in dev]]).score
    print(
        f"best step {best_step}, dev loss {best_loss:.4f}, dev BLEU (greedy) {bleu:.2f}"
    )

    info = {
        **vars(args),
        "n_train": len(train),
        "best_step": best_step,
        "best_dev_loss": best_loss,
        "dev_bleu_greedy": bleu,
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    ARGS = parse_args()
    main()
