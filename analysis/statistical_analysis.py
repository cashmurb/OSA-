import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import json
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

PRED_FILES = {
    "M1": "experiments/finetune_M1_v4/m1_test_preds.csv",
    "M2": "experiments/finetune_M2_v4/m2_test_preds.csv",
    "M3": "experiments/finetune_M3_v2/m3_test_preds.csv",
    "M4": "experiments/finetune_M4_v1/m4_test_preds.csv",
}

THRESHOLDS_PATH = "analysis/config_thresholds.json"
N_BOOTSTRAP = 1000
RANDOM_SEED = 42


def load_thresholds(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Threshold config not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_preds(path):
    idxs, y, p = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            idxs.append(int(row["idx"]) if "idx" in row and row["idx"] != "" else None)
            y.append(int(float(row["label"])))
            p.append(float(row["prob"]))
    return np.array(idxs, dtype=object), np.array(y, dtype=int), np.array(p, dtype=float)


def assert_pair_alignment(name_a, idx_a, y_a, name_b, idx_b, y_b):
    if len(y_a) != len(y_b):
        raise ValueError(f"Length mismatch: {name_a} vs {name_b}")
    if not np.array_equal(y_a, y_b):
        raise ValueError(f"Label mismatch: {name_a} vs {name_b}")
    if all(v is not None for v in idx_a) and all(v is not None for v in idx_b):
        if not np.array_equal(idx_a, idx_b):
            raise ValueError(f"Idx mismatch: {name_a} vs {name_b}")


def bootstrap_ci(y, p, threshold, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(y)
    aurocs, pr_aucs, f1s = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = y[idx], p[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aurocs.append(roc_auc_score(y_b, p_b))
        pr_aucs.append(average_precision_score(y_b, p_b))
        f1s.append(f1_score(y_b, (p_b >= threshold).astype(int), zero_division=0))
    lo, hi = alpha / 2, 1 - alpha / 2
    return {
        "auroc": (np.percentile(aurocs, lo*100), np.percentile(aurocs, hi*100), np.mean(aurocs)),
        "pr_auc": (np.percentile(pr_aucs, lo*100), np.percentile(pr_aucs, hi*100), np.mean(pr_aucs)),
        "f1": (np.percentile(f1s, lo*100), np.percentile(f1s, hi*100), np.mean(f1s)),
    }


def mcnemar_test(y_true, p1, p2, th1, th2):
    pred1 = (p1 >= th1).astype(int)
    pred2 = (p2 >= th2).astype(int)
    correct1 = pred1 == y_true
    correct2 = pred2 == y_true
    b = int((correct1 & ~correct2).sum())
    c = int((~correct1 & correct2).sum())
    if (b + c) == 0:
        return b, c, float("nan"), float("nan")
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return b, c, chi2, p_value


def main():
    np.random.seed(RANDOM_SEED)
    thresholds_cfg = load_thresholds(THRESHOLDS_PATH)

    print("Loading predictions...")
    data = {}
    for name, path in PRED_FILES.items():
        if not os.path.isfile(path):
            print(f"  [SKIP] {name}: {path} not found")
            continue
        idx, y, p = load_preds(path)
        if name not in thresholds_cfg:
            raise KeyError(f"Missing threshold for {name} in {THRESHOLDS_PATH}")
        th = float(thresholds_cfg[name])
        data[name] = {"idx": idx, "y": y, "p": p, "threshold": th}
        print(f"  {name}: {len(y)} clips  (pos={y.sum()}  neg={(y==0).sum()})  threshold={th:.3f}")

    print(f"\nRunning {N_BOOTSTRAP} bootstrap iterations per variant...")
    ci_results = {}
    for name, d in data.items():
        print(f"  Bootstrapping {name}...")
        ci_results[name] = bootstrap_ci(d["y"], d["p"], d["threshold"])

    print("\nRunning McNemar's test: M2 vs M3...")
    if "M2" in data and "M3" in data:
        assert_pair_alignment("M2", data["M2"]["idx"], data["M2"]["y"],
                              "M3",  data["M3"]["idx"],  data["M3"]["y"])
        b, c, chi2, p_val = mcnemar_test(
            data["M2"]["y"],
            data["M2"]["p"], data["M3"]["p"],
            data["M2"]["threshold"], data["M3"]["threshold"],
        )
    else:
        b = c = chi2 = p_val = None

    print("\n")
    print("═" * 72)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("═" * 72)
    print(f"{'Variant':<10} {'AUROC':>22} {'PR-AUC':>22} {'F1':>22}")
    print("─" * 72)
    for name, ci in ci_results.items():
        auroc_str = f"{ci['auroc'][2]:.3f} ({ci['auroc'][0]:.3f}–{ci['auroc'][1]:.3f})"
        pr_str    = f"{ci['pr_auc'][2]:.3f} ({ci['pr_auc'][0]:.3f}–{ci['pr_auc'][1]:.3f})"
        f1_str    = f"{ci['f1'][2]:.3f} ({ci['f1'][0]:.3f}–{ci['f1'][1]:.3f})"
        print(f"{name:<10} {auroc_str:>22} {pr_str:>22} {f1_str:>22}")
    print("─" * 72)
    print("Format: mean (lower–upper), 95% CI via bootstrap (n=1000)")

    if p_val is not None:
        print("\n")
        print("═" * 72)
        print(" McNEMAR'S TEST — M2 vs M3")
        print("═" * 72)
        print(f" M2 threshold: {data['M2']['threshold']:.3f}")
        print(f" M3 threshold: {data['M3']['threshold']:.3f}")
        print(f" b (M2✓, M3✗): {b}")
        print(f" c (M2✗, M3✓): {c}")
        print(f" χ² statistic: {chi2:.4f}")
        print(f" P-value: {p_val:.4f}")
        sig = "p < 0.001" if p_val < 0.001 else (f"p = {p_val:.4f} (significant)" if p_val < 0.05 else f"p = {p_val:.4f} (not significant)")
        print(f"  Result: {sig}")
        print("─" * 72)

    out = {
        "bootstrap_ci": {
            name: {
                metric: {"mean": float(vals[2]), "lower": float(vals[0]), "upper": float(vals[1])}
                for metric, vals in ci.items()
            }
            for name, ci in ci_results.items()
        },
        "thresholds": {name: float(d["threshold"]) for name, d in data.items()},
        "mcnemar": {"b": b, "c": c,
                    "chi2":    float(chi2)    if chi2    is not None else None,
                    "p_value": float(p_val) if p_val is not None else None},
    }

    out_path = "analysis/figures/statistical_analysis.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
