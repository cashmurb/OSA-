import os
import csv
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["ablation", "baseline"],
    default="ablation",
    help="ablation = OSA variants only | baseline = all models including baselines"
)
args = parser.parse_args()

# Model sets
MODELS_ABLATION = {
    "M1": "experiments/finetune_M1_v4/m1_test_preds.csv",
    "M2": "experiments/finetune_M2_v4/m2_test_preds.csv",
    "M3": "experiments/finetune_M3_v2/m3_test_preds.csv",
    "M4": "experiments/finetune_M4_v1/m4_test_preds.csv",
    "M5": "experiments/finetune_M5_v1/m5_test_preds.csv",
}

MODELS_BASELINE = {
    "ResNet-50": "experiments/baseline_resnet50/resnet50_test_preds.csv",
    "R3D-18": "experiments/baseline_r3d18/r3d18_test_preds.csv",
    "M1": "experiments/finetune_M1_v4/m1_test_preds.csv",
    "M2": "experiments/finetune_M2_v4/m2_test_preds.csv",
    "M3": "experiments/finetune_M3_v2/m3_test_preds.csv",
    "M4": "experiments/finetune_M4_v1/m4_test_preds.csv",
    "M5": "experiments/finetune_M5_v1/m5_test_preds.csv",
}

MODELS = MODELS_ABLATION if args.mode == "ablation" else MODELS_BASELINE

THRESHOLDS_PATH = "analysis/config_thresholds.json"
OUT_DIR = (
    "analysis/figures_ablation"
    if args.mode == "ablation"
    else "analysis/figures_baseline"
)
os.makedirs(OUT_DIR, exist_ok=True)

N_BOOT   = 1000
RNG_SEED = 42

COLORS = {
    "ResNet-50": "#808080",
    "R3D-18": "#404040",
    "M1": "#4878CF",
    "M2": "#6ACC65",
    "M3": "#E07B39",
    "M4": "#D65F5F",
    "M5": "#B47CC7",
}

STYLES = {
    "ResNet-50": "--",
    "R3D-18": "--",
    "M1": "-",
    "M2": "-",
    "M3": "-",
    "M4": "-.",
    "M5": ":",
}

# Helpers 
def load_thresholds(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Threshold config not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_preds(path):
    idxs, paths, y, p = [], [], [], []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            idxs.append(
                int(row["idx"])
                if "idx" in row and row["idx"] != ""
                else None
            )
            paths.append(row.get("path", ""))
            y.append(int(float(row["label"])))
            p.append(float(row["prob"]))
    return {
        "idx": np.array(idxs,  dtype=object),
        "path": np.array(paths, dtype=object),
        "y": np.array(y,     dtype=int),
        "p": np.array(p,     dtype=float),
    }


def assert_alignment(data):
    names = list(data.keys())
    if len(names) < 2:
        return
    ref_name = names[0]
    ref = data[ref_name]
    for name in names[1:]:
        cur = data[name]
        if len(ref["y"]) != len(cur["y"]):
            raise ValueError(
                f"Length mismatch: {ref_name}={len(ref['y'])} "
                f"vs {name}={len(cur['y'])}"
            )
        if not np.array_equal(ref["y"], cur["y"]):
            raise ValueError(
                f"Label order mismatch: {ref_name} vs {name}"
            )
        ref_has = all(v is not None for v in ref["idx"])
        cur_has = all(v is not None for v in cur["idx"])
        if ref_has and cur_has and not np.array_equal(ref["idx"], cur["idx"]):
            raise ValueError(
                f"Sample idx mismatch: {ref_name} vs {name}"
            )


def compute_metrics(y, p, th):
    y_hat = (p >= th).astype(int)
    tp = np.sum((y == 1) & (y_hat == 1))
    fp = np.sum((y == 0) & (y_hat == 1))
    tn = np.sum((y == 0) & (y_hat == 0))
    fn = np.sum((y == 1) & (y_hat == 0))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
    accuracy  = (
        (tp + tn) / (tp + tn + fp + fn)
        if (tp + tn + fp + fn) else 0.0
    )
    return {
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "accuracy": float(accuracy),
        "tp": int(tp), "fp": int(fp), "tn": int(tn),"fn": int(fn),
    }


def bootstrap_ci(y, p, th, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs, prs, f1s = [], [], []
    for _ in range(n_boot):
        idx     = rng.choice(n, size=n, replace=True)
        yt, yp  = y[idx], p[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
        prs.append(average_precision_score(yt, yp))
        f1s.append(
            f1_score(yt, (yp >= th).astype(int), zero_division=0)
        )

    def ci(arr):
        if len(arr) == 0:
            return (float("nan"), float("nan"))
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return float(lo), float(hi)

    return {
        "auroc_ci": ci(aucs),
        "pr_auc_ci": ci(prs),
        "f1_ci": ci(f1s),
    }


def fmt_ci(mean, ci_tuple):
    return f"{mean:.3f} [{ci_tuple[0]:.3f}, {ci_tuple[1]:.3f}]"


# Main 
def main():
    print(f"\n[plot_results] Mode: {args.mode.upper()}")
    print(f"[plot_results] Output: {OUT_DIR}/")

    thresholds_cfg = load_thresholds(THRESHOLDS_PATH)

    data = {}
    for name, path in MODELS.items():
        if not os.path.isfile(path):
            print(f"  WARNING: {path} not found — skipping {name}.")
            continue
        data[name] = load_preds(path)
        y = data[name]["y"]
        print(
            f"  Loaded {name}: {len(y)} clips "
            f"(pos={y.sum()} neg={(y==0).sum()})"
        )

    if not data:
        raise RuntimeError("No prediction CSVs found.")

    assert_alignment(data)

    first_name = list(data.keys())[0]
    prevalence = float(data[first_name]["y"].mean())

    # Compute metrics 
    results = {}
    for name, d in data.items():
        if name not in thresholds_cfg:
            raise KeyError(
                f"Missing threshold for '{name}' in {THRESHOLDS_PATH}"
            )
        y, p   = d["y"], d["p"]
        th     = float(thresholds_cfg[name])
        bm     = compute_metrics(y, p, th)
        auroc  = (
            roc_auc_score(y, p)
            if len(np.unique(y)) > 1 else float("nan")
        )
        pr_auc = (
            average_precision_score(y, p)
            if len(np.unique(y)) > 1 else float("nan")
        )
        ci_    = bootstrap_ci(y, p, th, n_boot=N_BOOT, seed=RNG_SEED)
        results[name] = {
            "threshold": th,
            "auroc": float(auroc),
            "pr_auc": float(pr_auc),
            "f1": bm["f1"],
            "precision": bm["precision"],
            "recall": bm["recall"],
            "accuracy": bm["accuracy"],
            "tp": bm["tp"], "fp": bm["fp"],
            "tn": bm["tn"], "fn": bm["fn"],
            "auroc_ci": ci_["auroc_ci"],
            "pr_auc_ci": ci_["pr_auc_ci"],
            "f1_ci": ci_["f1_ci"],
        }

    out_json = os.path.join(OUT_DIR, "results_with_bootstrap_ci.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_json}")

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, d in data.items():
        fpr, tpr, _ = roc_curve(d["y"], d["p"])
        ax.plot(
            fpr, tpr,
            color=COLORS[name],
            linestyle=STYLES[name],
            linewidth=2,
            label=f"{name} (AUC={results[name]['auroc']:.3f})",
        )
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve", fontsize=12)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    roc_path = os.path.join(OUT_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[saved] {roc_path}")

    # PR curve 
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, d in data.items():
        prec, rec, _ = precision_recall_curve(d["y"], d["p"])
        ax.plot(
            rec, prec,
            color=COLORS[name],
            linestyle=STYLES[name],
            linewidth=2,
            label=f"{name} (AP={results[name]['pr_auc']:.3f})",
        )
    ax.axhline(
        prevalence, color="gray", lw=1, linestyle=":",
        alpha=0.8, label=f"Prevalence ({prevalence:.3f})",
    )
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision\u2013Recall Curve", fontsize=12)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    pr_path = os.path.join(OUT_DIR, "pr_curve.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()
    print(f"[saved] {pr_path}")

    # F1 vs threshold 
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, d in data.items():
        y, p = d["y"], d["p"]
        thresholds = np.linspace(0.0, 1.0, 500)
        f1s = [
            f1_score(y, (p >= th).astype(int), zero_division=0)
            for th in thresholds
        ]
        ax.plot(
            thresholds, f1s,
            lw=2,
            color=COLORS[name],
            linestyle=STYLES[name],
            label=f"{name} (fixed={results[name]['threshold']:.3f})",
        )
        ax.axvline(
            results[name]["threshold"],
            color=COLORS[name], linestyle=":", alpha=0.5
        )
    ax.set_xlabel("Threshold", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("F1 vs Threshold", fontsize=12)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    f1_path = os.path.join(OUT_DIR, "f1_vs_threshold.png")
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print(f"[saved] {f1_path}")

    # Ablation bar chart with CI 
    metrics_labels = [
        ("auroc", "AUROC", "auroc_ci"),
        ("pr_auc", "PR-AUC", "pr_auc_ci"),
        ("f1", "Calibrated F1", "f1_ci"),
    ]
    variants = list(results.keys())
    n_variants = len(variants)
    x = np.arange(n_variants)
    bar_colors = [COLORS[v] for v in variants]

    # Wider figure for 7-model baseline mode
    fig_width = 14 if n_variants <= 5 else 18
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, 4.5))

    for ax, (metric_key, metric_label, ci_key) in zip(axes, metrics_labels):
        vals = [results[v][metric_key] for v in variants]
        cis = [results[v][ci_key]     for v in variants]
        yerr_lower = [v - ci[0] for v, ci in zip(vals, cis)]
        yerr_upper = [ci[1] - v for v, ci in zip(vals, cis)]
        yerr = np.array([yerr_lower, yerr_upper])

        bars = ax.bar(
            x, vals, 0.6,
            color=bar_colors, edgecolor="black", linewidth=0.7,
            yerr=yerr, capsize=4, ecolor="black",
        )
        for b, v, ci in zip(bars, vals, cis):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.015,
                f"{v:.3f}\n[{ci[0]:.3f},{ci[1]:.3f}]",
                ha="center", va="bottom",
                fontsize=7 if n_variants > 5 else 8,
            )

        ax.set_xticks(x)
        # Rotate labels for baseline mode to prevent overlap
        ax.set_xticklabels(
            variants,
            fontsize=9 if n_variants > 5 else 10,
            rotation=30 if n_variants > 5 else 0,
            ha="right" if n_variants > 5 else "center",
        )
        ax.set_ylabel(metric_label, fontsize=10)
        ax.set_ylim(0, 1.20)
        ax.set_title(metric_label, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    title = (
        "Ablation Performance with 95% Bootstrap CI"
        if args.mode == "ablation"
        else "Baseline and Ablation Performance with 95% Bootstrap CI"
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    bar_path = os.path.join(OUT_DIR, "ablation_bar_with_ci.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[saved] {bar_path}")

    # Summary text 
    txt_path = os.path.join(OUT_DIR, "results_summary.txt")
    with open(txt_path, "w") as f:
        f.write(
            "Variant\tAUROC [95% CI]\tPR-AUC [95% CI]\t"
            "F1 [95% CI]\tThreshold\n"
        )
        for name in variants:
            r = results[name]
            f.write(
                f"{name}\t"
                f"{fmt_ci(r['auroc'], r['auroc_ci'])}\t"
                f"{fmt_ci(r['pr_auc'], r['pr_auc_ci'])}\t"
                f"{fmt_ci(r['f1'], r['f1_ci'])}\t"
                f"{r['threshold']:.3f}\n"
            )
    print(f"[saved] {txt_path}")

    # Console summary 
    print(
        f"\n Results Summary ({args.mode.upper()}) "
        + "─" * 50
    )
    print(
        f"{'Variant':<14} {'AUROC [95% CI]':<28} "
        f"{'PR-AUC [95% CI]':<28} {'F1 [95% CI]':<28} {'Thr':>6}"
    )
    print("-" * 110)
    for name in variants:
        r = results[name]
        print(
            f"{name:<14} "
            f"{fmt_ci(r['auroc'],  r['auroc_ci']):<28} "
            f"{fmt_ci(r['pr_auc'], r['pr_auc_ci']):<28} "
            f"{fmt_ci(r['f1'],     r['f1_ci']):<28} "
            f"{r['threshold']:>6.3f}"
        )
    print("-" * 110)
    print(f"\nAll figures saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()