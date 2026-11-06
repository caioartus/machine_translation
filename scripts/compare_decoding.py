import argparse
import contextlib
import json
import time
from pathlib import Path

import torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import M2M100ForConditionalGeneration

from data_utils import get_tokenizer, read_jsonl
from train import translate

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--out_dir", required=True)
args = p.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("GPU unavailable")

run = Path(args.run)
checkpoint = run / "best"
if not checkpoint.is_dir():
    raise FileNotFoundError(checkpoint)

dev = read_jsonl("data/processed/alma_dev.jsonl")
sources = [r["zh"] for r in dev]
references = [[r["en"] for r in dev]]

tokenizer = get_tokenizer()
tokenizer.src_lang = "zh"
tokenizer.tgt_lang = "en"

model = M2M100ForConditionalGeneration.from_pretrained(
    checkpoint, local_files_only=True
).to("cuda")
model.eval()

# 两种解码仅改变 num_beams。
model.generation_config.do_sample = False
model.generation_config.num_return_sequences = 1
model.generation_config.length_penalty = 1.0
model.generation_config.early_stopping = False

autocast = (
    lambda: torch.autocast("cuda", dtype=torch.bfloat16)
) if torch.cuda.is_bf16_supported() else contextlib.nullcontext

out = Path(args.out_dir) / run.name
out.mkdir(parents=True, exist_ok=False)
results = []

for beams in (1, 4):
    torch.cuda.synchronize()
    started = time.perf_counter()
    hyps = translate(
        model, tokenizer, sources, "cuda", autocast,
        batch_size=16, num_beams=beams
    )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started

    bleu = BLEU(tokenize="13a")
    chrf = CHRF()
    result = {
        "run": str(run),
        "num_beams": beams,
        "n_dev": len(dev),
        "bleu": bleu.corpus_score(hyps, references).score,
        "chrf": chrf.corpus_score(hyps, references).score,
        "generation_seconds": seconds,
        "bleu_signature": str(bleu.get_signature()),
        "chrf_signature": str(chrf.get_signature()),
        "batch_size": 16,
        "length_penalty": 1.0,
        "early_stopping": False,
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    results.append(result)

    with (out / f"beam{beams}.jsonl").open("w") as f:
        for row, hyp in zip(dev, hyps):
            f.write(json.dumps({
                "id": row["id"], "zh": row["zh"],
                "ref": row["en"], "hyp": hyp
            }, ensure_ascii=False) + "\n")

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(result), flush=True)
