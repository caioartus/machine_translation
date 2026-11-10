"""Plan, execute and audit the ALMA official-test evaluation matrix."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
WORK = ROOT / "results/_evaluation_workflow"
PLAN = WORK / "plan.json"
SETS = {"wmt22": 1875, "tatoeba": 469}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def target(task):
    return ROOT / task["output"]


def validate(task, need_comet=False):
    from sacrebleu.metrics import BLEU, CHRF

    out = target(task)
    data = ROOT / task["data"]
    source = rows(data)
    predictions = rows(out / "predictions.jsonl")
    meta = read(out / "metadata.json")
    metrics = read(out / "metrics.json")
    n = SETS[task["test_set"]]

    require(len(source) == len(predictions) == n, "Wrong sentence count")
    require(len({r["id"] for r in predictions}) == n, "Duplicate prediction IDs")
    expected = {
        "status": "complete", "partial_dataset": False,
        "n_total": n, "n_evaluated": n,
        "model": task["model"], "data_sha256": task["data_sha256"],
        "num_beams": task["beam"], "batch_size": 16,
        "max_length": 256, "reference_field": "en",
        "do_sample": False, "num_return_sequences": 1,
        "length_penalty": 1.0, "early_stopping": False,
    }
    for key, value in expected.items():
        require(meta.get(key) == value, f"Metadata mismatch: {key}")
    require(sha(data) == task["data_sha256"], "Processed data changed")

    for original, prediction in zip(source, predictions):
        require(
            all(original[k] == prediction[k] for k in ("id", "zh", "en")),
            "Source/reference/ID/order mismatch",
        )
        require(isinstance(prediction["prediction"], str), "Invalid prediction")

    hypotheses = [r["prediction"] for r in predictions]
    references = [[r["en"] for r in source]]
    for name, scorer in (("bleu", BLEU(tokenize="13a")), ("chrf", CHRF())):
        recalculated = scorer.corpus_score(hypotheses, references).score
        saved = float(metrics[name]["score"])
        require(math.isfinite(saved), f"Nonfinite {name}")
        require(abs(saved - recalculated) < 1e-8, f"{name} does not reproduce")
        require(
            metrics[name]["signature"] == str(scorer.get_signature()),
            f"{name} signature/version mismatch",
        )

    record = {
        "dataset": "alma", "condition": task["condition"],
        "size": task["size"], "seed": task["seed"],
        "beam": task["beam"], "test_set": task["test_set"], "n": n,
        "bleu": metrics["bleu"]["score"], "chrf": metrics["chrf"]["score"],
        "comet": "", "empty": sum(not h.strip() for h in hypotheses),
        "source_truncation_count": meta["source_truncation_count"],
        "output": task["output"],
    }

    comet_path = out / "comet.json"
    if comet_path.exists():
        comet = read(comet_path)
        scores = comet["sentence_scores"]
        require(comet["model"] == "Unbabel/wmt22-comet-da", "Wrong COMET model")
        require(comet["n_evaluated"] == len(scores) == n, "COMET count mismatch")
        require(
            [r["id"] for r in scores] == [r["id"] for r in predictions],
            "COMET IDs/order mismatch",
        )
        require(
            comet["predictions_sha256"] == sha(out / "predictions.jsonl"),
            "COMET prediction hash mismatch",
        )
        values = [float(r["score"]) for r in scores]
        score = float(comet["system_score"])
        require(all(math.isfinite(x) for x in values + [score]), "Nonfinite COMET")
        require(
            math.isclose(score, sum(values) / n, rel_tol=1e-6, abs_tol=1e-6),
            "COMET mean mismatch",
        )
        record["comet"] = score
    elif need_comet:
        raise ValueError("COMET missing")

    return record


def make_plan():
    require(not PLAN.exists(), "Plan already exists; do not recreate it.")
    require(os.access(os.environ["COMET_PYTHON"], os.X_OK), "COMET Python unavailable")
    require(Path(os.environ["HF_HOME"]).is_dir(), "COMET cache unavailable")

    tasks = []
    training_records = {}
    settings = {
        "1000": (3e-6, 300, 30, 10),
        "2000": (3e-6, 375, 38, 20),
        "5000": (3e-6, 940, 94, 20),
        "full": (1e-5, 2900, 290, 100),
    }

    for condition in ("uniform", "length", "joint", "loss"):
        owner = "c/caioart" if condition == "loss" else "s/simonwim"
        base = Path(f"/mnt/scratch/{owner}/shared/runs")
        for size, (lr, steps, warmup, every) in settings.items():
            for seed in range(3):
                run = f"{condition}_n{size}_seed{seed}"
                info_path = base / run / "run_info.json"
                info = read(info_path)
                expected = {
                    "condition": condition, "seed": seed, "steps": steps,
                    "batch_size": 16, "lr": lr, "cap": 3.0,
                    "warmup_steps": warmup, "eval_every": every,
                    "weight_decay": 0.01,
                }
                require(str(info["size"]) == size, f"{run}: wrong size")
                require(info.get("dataset", "alma") == "alma", f"{run}: wrong dataset")
                for key, value in expected.items():
                    require(info.get(key) == value, f"{run}: unexpected {key}")

                model = base / run / "best"
                for name in ("model.safetensors", "config.json", "sentencepiece.bpe.model"):
                    require(os.access(model / name, os.R_OK), f"Unreadable: {model / name}")

                destination = ROOT / "results/alma" / run / "run_info.json"
                if destination.exists():
                    require(read(destination) == info, f"Conflicting record: {destination}")
                training_records[str(destination)] = info

                for beam in (1, 4):
                    for split in SETS:
                        data = f"data/processed/{split}_test.jsonl"
                        tasks.append({
                            "run": run, "condition": condition, "size": size,
                            "seed": seed, "beam": beam, "test_set": split,
                            "model": str(model), "data": data,
                            "data_sha256": sha(data),
                            "run_info_sha256": sha(info_path),
                            "output": f"results/alma/{run}/beam{beam}/{split}",
                        })

    pending = []
    translation_done = comet_done = 0
    for i, task in enumerate(tasks):
        if target(task).exists():
            validate(task)
            translation_done += 1
            if (target(task) / "comet.json").exists():
                comet_done += 1
            else:
                pending.append(i)
        else:
            pending.append(i)

    # Freeze a copy of the implementation used by the whole batch.
    require(
        "model.generation_config.early_stopping = False"
        in (ROOT / "scripts/evaluate.py").read_text(),
        "Explicit decoding fix is absent",
    )
    snapshot = WORK / "code"
    snapshot.mkdir(parents=True, exist_ok=False)
    names = ("evaluate.py", "score_comet.py", "train.py", "data_utils.py")
    for name in names:
        shutil.copyfile(ROOT / "scripts" / name, snapshot / name)

    for path, info in training_records.items():
        if not Path(path).exists():
            save_new(path, info)

    save_new(PLAN, {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "code_sha256": {name: sha(snapshot / name) for name in names},
        "tasks": tasks,
    })
    (WORK / "pending.txt").write_text(
        ",".join(map(str, pending)) + "\n", encoding="utf-8"
    )
    print(f"Planned: {len(tasks)} combinations")
    print(f"Validated existing translations: {translation_done}")
    print(f"Validated existing COMET results: {comet_done}")
    print(f"Missing translations: {len(tasks) - translation_done}")
    print(f"Missing COMET results: {len(tasks) - comet_done}")
    print(f"Pending array tasks: {len(pending)}")


def execute(index):
    plan = read(PLAN)
    task = plan["tasks"][index]
    snapshot = WORK / "code"
    for name, digest in plan["code_sha256"].items():
        require(sha(snapshot / name) == digest, f"Frozen code changed: {name}")
    require(sha(task["data"]) == task["data_sha256"], "Data changed")
    info_path = Path(task["model"]).parent / "run_info.json"
    require(sha(info_path) == task["run_info_sha256"], "Training record changed")

    out = target(task)
    print(f"START {index}: {task['output']}", flush=True)
    if not out.exists():
        subprocess.run([
            sys.executable, str(snapshot / "evaluate.py"),
            "--model", task["model"], "--data", task["data"],
            "--num_beams", str(task["beam"]), "--batch_size", "16",
            "--device", "cuda", "--out_dir", str(out),
        ], check=True)

    validate(task)
    if not (out / "comet.json").exists():
        subprocess.run([
            os.environ["COMET_PYTHON"], str(snapshot / "score_comet.py"),
            "--predictions", str(out / "predictions.jsonl"),
            "--output", str(out / "comet.json"),
            "--batch_size", "8", "--gpus", "1",
        ], check=True)

    validate(task, need_comet=True)
    print(f"PASS {index}: {task['output']}", flush=True)


def audit():
    plan = read(PLAN)
    records, failures = [], []
    for i, task in enumerate(plan["tasks"]):
        try:
            records.append(validate(task, need_comet=True))
            print(f"PASS {i}: {task['output']}")
        except Exception as exc:
            failures.append(i)
            print(f"FAIL {i}: {task['output']}: {exc}")

    (WORK / "failed.txt").write_text(
        ",".join(map(str, failures)) + "\n", encoding="utf-8"
    )
    summary = WORK / ("summary.partial.csv" if failures else "summary.csv")
    if records:
        with summary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(f"\nValidated {len(records)}/{len(plan['tasks'])}; summary: {summary}")
    if failures:
        raise SystemExit(1)


parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("plan", "task", "audit"))
parser.add_argument("index", nargs="?", type=int)
args = parser.parse_args()
if args.mode == "plan":
    make_plan()
elif args.mode == "task":
    require(args.index is not None, "Task index required")
    execute(args.index)
else:
    audit()
