"""Smallest useful fine-tuning loop: M2M-100 (zh -> en) on a slice of ALMA-Human-Parallel."""
import argparse

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--n_train", type=int, default=2000, help="number of training pairs to use")
parser.add_argument("--n_val", type=int, default=200, help="number of validation pairs to use")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--output_dir", default="outputs/test_run")
args = parser.parse_args()

MODEL_NAME = "facebook/m2m100_418M"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

# --- Model and tokenizer ---------------------------------------------------
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.src_lang, tokenizer.tgt_lang = "zh", "en"
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

# --- Data: the zh-en subset has a "train" and a "validation" split ----------
# Each row looks like {"translation": {"zh": "...", "en": "..."}}.
data = load_dataset("haoranxu/ALMA-Human-Parallel", "zh-en")
train_data = data["train"].shuffle(seed=0).select(range(args.n_train))
val_data = data["validation"].select(range(args.n_val))


def collate(rows):
    """Turn a list of rows into one padded batch of tensors."""
    batch = tokenizer(
        [r["translation"]["zh"] for r in rows],
        text_target=[r["translation"]["en"] for r in rows],  # becomes batch["labels"]
        # 256 covers every cleaned ALMA pair (prepare_alma.py drops the 8 longer ones),
        # so truncation never cuts a target sentence short.
        max_length=256, truncation=True, padding=True, return_tensors="pt",
    )
    # Padding positions must not count towards the loss: -100 is ignored by the loss.
    batch["labels"][batch["labels"] == tokenizer.pad_token_id] = -100
    return batch


train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_data, batch_size=args.batch_size * 2, collate_fn=collate)


def val_loss():
    model.eval()  # turns dropout off
    total, n = 0.0, 0
    with torch.no_grad():  # no gradients needed, saves memory
        for batch in val_loader:
            batch = batch.to(device)
            total += model(**batch).loss.item() * len(batch["input_ids"])
            n += len(batch["input_ids"])
    model.train()
    return total / n


def translate(sentences):
    model.eval()
    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id("en"),
                             num_beams=1, max_new_tokens=100)
    model.train()
    return tokenizer.batch_decode(out, skip_special_tokens=True)


# --- Training ---------------------------------------------------------------
test_sentences = ["我喜欢学习机器翻译。", "今天天气很好，我们去公园散步吧。"]
print("before:", translate(test_sentences))
print(f"val loss before: {val_loss():.4f}")

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
model.train()  # turns dropout on
step = 0
for epoch in range(args.epochs):
    for batch in train_loader:
        batch = batch.to(device)
        loss = model(**batch).loss  # passing labels makes the model compute cross-entropy
        loss.backward()             # compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoid huge updates
        optimizer.step()            # update weights
        optimizer.zero_grad()       # reset gradients for the next step
        step += 1
        if step % 10 == 0:
            print(f"epoch {epoch} step {step} train loss {loss.item():.4f}", flush=True)

print(f"val loss after: {val_loss():.4f}")
print("after:", translate(test_sentences))

model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print("saved to", args.output_dir)
