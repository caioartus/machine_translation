"""Plot the dev-loss curves of the learning-rate tuning runs, for the README.

Reads the committed tuning results (runs_tune2/, runs_tune_5000/, runs_tune_full/, runs_tune_full2/)
and writes two figures to figures/:

    tuning_dev_loss.png     uniform, one panel per size / budget, one line per learning rate
    tuning_1000_conditions.png   size 1000, one panel per sampling condition

Run from the repository root (CPU only, a few seconds):
    python scripts/plot_tuning.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path("figures")

# One clearly distinct colour per learning rate, the same in every panel.
LR_COLOURS = {3e-6: "#e07b39", 5e-6: "#9a9a9a", 1e-5: "#2a78d6", 3e-5: "#6b3fa0"}


def load_runs(tune_dir, condition):
    """Return [(run_info, dev_curve)] for one condition in a tuning folder, sorted by lr."""
    runs = []
    for info_file in Path(tune_dir).glob(f"lr*/{condition}_n*_seed0/run_info.json"):
        info = json.loads(info_file.read_text())
        runs.append((info, pd.read_csv(info_file.parent / "dev_curve.csv")))
    return sorted(runs, key=lambda run: run[0]["lr"])


def plot_panel(ax, runs, title, chosen_lr=None):
    """One line per learning rate; a dot at the selected checkpoint (lowest dev loss)."""
    for info, curve in runs:
        lr = info["lr"]
        chosen = lr == chosen_lr
        label = f"lr {lr:g}" + (" (chosen)" if chosen else "")
        ax.plot(curve["step"], curve["dev_loss"], color=LR_COLOURS[lr], linewidth=2.5 if chosen else 1.6,
                label=label)
        ax.plot(info["best_step"], info["best_dev_loss"], "o", color=LR_COLOURS[lr], markersize=7,
                markeredgecolor="white", markeredgewidth=1.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # Figure 1: uniform, every size / budget that was tuned.
    panels = [
        ("runs_tune2", "size 1000, 300 steps", 3e-6),
        ("runs_tune_5000", "size 5000, 940 steps", 3e-6),
        ("runs_tune_full", "full, 2900 steps", 1e-5),
        ("runs_tune_full2", "full, 4800 steps (budget check)", None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 3.8))
    for ax, (tune_dir, title, chosen_lr) in zip(axes, panels):
        plot_panel(ax, load_runs(tune_dir, "uniform"), title, chosen_lr)
    axes[0].set_ylabel("dev loss (alma_dev)")
    fig.suptitle("Learning-rate tuning, uniform sampling: dev loss (lower is better; dot = selected checkpoint)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tuning_dev_loss.png", dpi=120)
    plt.close(fig)

    # Figure 2: size 1000, the three conditions (they all pick the same learning rate).
    conditions = ["uniform", "length", "joint"]
    fig, axes = plt.subplots(1, len(conditions), figsize=(13, 3.8), sharey=True)
    for ax, condition in zip(axes, conditions):
        plot_panel(ax, load_runs("runs_tune2", condition), condition, chosen_lr=3e-6)
    axes[0].set_ylabel("dev loss (alma_dev)")
    fig.suptitle("Size 1000: dev loss per sampling condition (all three pick lr 3e-6)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tuning_1000_conditions.png", dpi=120)
    plt.close(fig)

    print(f"wrote {OUT_DIR}/tuning_dev_loss.png and {OUT_DIR}/tuning_1000_conditions.png")


if __name__ == "__main__":
    main()
