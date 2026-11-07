"""Validated joins and ALMA-relative coverage shared by error-analysis tools."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from data_utils import LENGTH_LABELS, RARITY_LABELS, all_cells


def unique_ids(df, label):
    if "id" not in df or df["id"].isna().any() or (df["id"].astype(str).str.strip() == "").any():
        raise ValueError(f"{label}: missing/empty sentence id")
    if df["id"].duplicated().any():
        raise ValueError(f"{label}: duplicate sentence ids")


def align_to(df, ids, label):
    unique_ids(df, label)
    if len(ids) != len(set(ids)):
        raise ValueError("reference ids are duplicated")
    if set(df["id"]) != set(ids):
        raise ValueError(f"{label}: sentence-id set differs (missing or extra sentences)")
    return df.set_index("id").loc[ids].reset_index()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frequencies(path):
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    if df["subword"].duplicated().any():
        raise ValueError("duplicate subword in ALMA frequency table")
    if (df["count"] <= 0).any() or (df["count"] % 1 != 0).any():
        raise ValueError("ALMA frequency counts must be positive integers")
    return Counter(dict(zip(df["subword"], df["count"].astype(int))))


def coverage_features(rows, counts, tokenizer, threshold=5, leave_one_out=False):
    if threshold < 1:
        raise ValueError("low-frequency threshold must be >= 1")
    result = []
    for row in rows:
        pieces = tokenizer.tokenize(row["zh"])
        own = Counter(pieces) if leave_one_out else Counter()
        observed = [counts[p] - own[p] for p in pieces]
        if any(c < 0 for c in observed):
            raise ValueError("frequency table does not match ALMA training sentences")
        n = len(pieces)
        unseen = sum(c == 0 for c in observed)
        low = sum(1 <= c <= threshold for c in observed)
        result.append({"id": row["id"], "coverage_src_len": n,
                       "n_unseen_alma": unseen, "n_lowfreq_alma": low,
                       "unseen_ratio": unseen / n if n else 0.0,
                       "lowfreq_ratio": low / n if n else 0.0,
                       "lowfreq_threshold": threshold,
                       "coverage_reference": "alma_train_leave_one_out" if leave_one_out else "alma_train"})
    df = pd.DataFrame(result)
    unique_ids(df, "coverage features")
    return df


def group_rows(df):
    yield "overall", "all", df
    for kind, col, labels in (("length", "length_bin", LENGTH_LABELS),
                              ("rarity", "rarity_bin", RARITY_LABELS),
                              ("cell", "cell", all_cells())):
        for label in labels:
            yield kind, label, df[df[col] == label]
    if "domain" in df:
        for label in sorted(df["domain"].fillna("unknown").astype(str).unique()):
            yield "domain", label, df[df["domain"].fillna("unknown").astype(str) == label]


def infer_run_metadata(name):
    match = re.search(r"(?:^|/)(uniform|length|joint)_n([^_/]+)_seed(\d+)(?:$|[_-])", name)
    info = {"condition": "pretrained" if name == "pretrained" else "unknown",
            "train_size": "unknown", "seed": "unknown", "num_beams": "unknown"}
    if match:
        info.update(condition=match[1], train_size=match[2], seed=int(match[3]))
    beam = re.search(r"(?:^|[_/-])beam(?:s)?(\d+)(?:$|[_/-])", name)
    if beam:
        info["num_beams"] = int(beam[1])
    return info


def paired_interval(diff, n_boot, seed):
    """Sentence-only, unadjusted percentile interval; bounded memory."""
    diff = np.asarray(diff, dtype=float)
    if len(diff) == 0 or not np.isfinite(diff).all() or n_boot < 1:
        raise ValueError("bootstrap needs finite nonempty differences and n_boot >= 1")
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, n_boot, 100):
        idx = rng.integers(0, len(diff), size=(min(100, n_boot - start), len(diff)))
        means.extend(diff[idx].mean(axis=1))
    return np.percentile(means, [2.5, 97.5])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
