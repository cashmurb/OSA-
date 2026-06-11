# DeLong's test for paired AUROC comparison: M2 vs M2_MTR
# Tests whether the AUROC difference is statistically significant.

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import numpy as np
from scipy import stats

# Load predictions 
def load_preds(path):
    y, p = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            y.append(int(float(row["label"])))
            p.append(float(row["prob"]))
    return np.array(y), np.array(p)

# DeLong's test implementation

def compute_midrank(x):
    """Compute midranks of x."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def compute_structural_components(y_true, y_pred):
    """
    Compute structural components V10 and V01 for DeLong's test.
    V10[i] = P(score of positive i > random negative)
    V01[j] = P(score of negative j < random positive)
    """
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    pos_scores = y_pred[pos_idx]
    neg_scores = y_pred[neg_idx]
    
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    
    # For each positive, compute fraction of negatives it outranks
    V10 = np.zeros(n_pos)
    for i, ps in enumerate(pos_scores):
        V10[i] = np.mean(ps > neg_scores) + 0.5 * np.mean(ps == neg_scores)
    
    # For each negative, compute fraction of positives that outrank it
    V01 = np.zeros(n_neg)
    for j, ns in enumerate(neg_scores):
        V01[j] = np.mean(pos_scores > ns) + 0.5 * np.mean(pos_scores == ns)
    
    return V10, V01

def delong_test(y_true, y_pred1, y_pred2):
    """
    DeLong's test for two correlated AUROCs on the same test set.
    Returns:
        auc1: AUROC of model 1
        auc2: AUROC of model 2
        z: z-statistic
        p_value: two-tailed p-value
        ci_diff: 95% confidence interval of (AUC2 - AUC1)
    """
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    
    V10_1, V01_1 = compute_structural_components(y_true, y_pred1)
    V10_2, V01_2 = compute_structural_components(y_true, y_pred2)
    
    auc1 = V10_1.mean()
    auc2 = V10_2.mean()
    
    # Covariance matrix of the two AUROCs
    # S = [[S11, S12], [S12, S22]]
    S11 = (np.var(V10_1, ddof=1) / n_pos + np.var(V01_1, ddof=1) / n_neg)
    
    S22 = (np.var(V10_2, ddof=1) / n_pos + np.var(V01_2, ddof=1) / n_neg)
    
    S12 = (np.cov(V10_1, V10_2, ddof=1)[0,1] / n_pos + np.cov(V01_1, V01_2, ddof=1)[0,1] / n_neg)
    
    # Variance of the difference
    var_diff = S11 + S22 - 2 * S12
    
    if var_diff <= 0:
        print("[WARN] Variance of difference is non-positive.")
        return auc1, auc2, 0.0, 1.0, (0.0, 0.0)
    
    # Z-statistic
    z = (auc2 - auc1) / np.sqrt(var_diff)
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # 95% confidence interval of the difference
    diff = auc2 - auc1
    margin = 1.96 * np.sqrt(var_diff)
    ci_diff = (diff - margin, diff + margin)
    
    return auc1, auc2, z, p_value, ci_diff


# Main 

def main():
    m2_csv = "experiments/finetune_M2_v4/m2_test_preds.csv"
    m2mtr_csv = "experiments/finetune_M3_v2/m3_test_preds.csv"
    
    print("Loading predictions...")
    y_m2, p_m2 = load_preds(m2_csv)
    y_m2mtr, p_m2mtr = load_preds(m2mtr_csv)
    
    # Sanity check 
        "Labels differ between M2 and M2_MTR. Always check that both used the same test split."
    
    print(f"Test set: {len(y_m2)} clips  "
          f"(positive={y_m2.sum()}  negative={(y_m2==0).sum()})")
    
    print("\nRunning DeLong's test: M2 vs M2_MTR...")
    auc_m2, auc_m2mtr, z, p, ci = delong_test(y_m2, p_m2, p_m2mtr)
    
    print("\n DeLong's Test Results ")
    print(f" M2 AUROC: {auc_m2:.4f}")
    print(f" M2_MTR AUROC: {auc_m2mtr:.4f}")
    print(f" Difference: {auc_m2mtr - auc_m2:+.4f}")
    print(f" 95% CI (diff): ({ci[0]:+.4f}, {ci[1]:+.4f})")
    print(f" Z-statistic: {z:.4f}")
    print(f" P-value: {p:.4f}")
  
    
    if p < 0.001:
        sig = "p < 0.001 (highly significant)"
    elif p < 0.01:
        sig = f"p = {p:.4f} (significant)"
    elif p < 0.05:
        sig = f"p = {p:.4f} (significant)"
    else:
        sig = f"p = {p:.4f} (not significant at α=0.05)"
    
    print(f"\n  Significance: {sig}")
    
    if p < 0.05:
        print(f"\n  CONCLUSION: M2_MTR significantly outperforms M2 in AUROC.")
        print(f"  The {auc_m2mtr - auc_m2:.3f} improvement is statistically significant.")
    else:
        print(f"\n  CONCLUSION: The AUROC difference is not statistically significant.")
    print()


if __name__ == "__main__":
    main()