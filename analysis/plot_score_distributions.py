# Predicted probability distributions for normal vs abnormal clips
# Shows score compression in spatial prior variants vs M2_MTR

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

MODELS = {
    "M1": "experiments/finetune_M1_v4/m1_test_preds.csv",
    "M2": "experiments/finetune_M2_v4/m2_test_preds.csv",
    "M3": "experiments/finetune_M3_v2/m3_test_preds.csv",
    "M4": "experiments/finetune_M4_v1/m4_test_preds.csv",
    "M5": "experiments/finetune_M5_v1/m5_test_preds.csv",
}

THRESHOLDS = {
    "M1": 0.271,
    "M2": 0.241,
    "M3": 0.055,
    "M4": 0.101,
    "M5": 0.087,
}

COLORS_POS = "#E07B39"
COLORS_NEG = "#4878CF"
OUT_DIR = "analysis/figures_ablation"
os.makedirs(OUT_DIR, exist_ok=True)


def load_preds(path):
    y, p = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            y.append(int(float(row["label"])))
            p.append(float(row["prob"]))
    return np.array(y), np.array(p)


def main():
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
    bins = np.linspace(0, 1, 41)

    for ax, (name, path) in zip(axes, MODELS.items()):
        y, p = load_preds(path)
        pos = p[y == 1]
        neg = p[y == 0]
        th = THRESHOLDS[name]

        ax.hist(neg, bins=bins, alpha=0.6, color=COLORS_NEG,
                density=True, label="Normal")
        ax.hist(pos, bins=bins, alpha=0.6, color=COLORS_POS,
                density=True, label="Abnormal")
        ax.axvline(th, color="black", linestyle="--",
                   linewidth=1.2, label=f"Thr={th:.3f}")

        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted probability", fontsize=9)
        ax.set_xlim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax == axes[0]:
            ax.set_ylabel("Density", fontsize=9)
        if ax == axes[-1]:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Predicted Probability Distributions: Normal vs Abnormal",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "score_distributions.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()