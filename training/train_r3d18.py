# R3D-18 baseline: 3D CNN with Kinetics-400 pretrained weights
# Temporal modelling via 3D convolutions — no TAH, no LoRA, no MTR

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "experiments/baseline_r3d18"
DATA_DIR = "data/echonet-pediatric"
LR = 3e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8
MAX_EPOCHS = 50
PATIENCE = 10
GRAD_CLIP = 0.5

os.makedirs(OUT_DIR, exist_ok=True)

# Uses the same dataset class as all OSA variants for fair comparison
from datasets.pediatric_dataset import EchoNetPediatricDataset as DS


# Model
class R3D18Baseline(nn.Module):
    def __init__(self, freeze_backbone=False):
        super().__init__()
        backbone = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)

        # Remove the original classifier
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(512, 1)

        # Conservative bias initialisation 
        nn.init.constant_(self.classifier.bias, -2.0)

        # Optional: freeze backbone and only train classifier
        # Set freeze_backbone=True to match OSA's frozen backbone approach
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x from dataset: (B, T, C, H, W)
        # R3D-18 expects: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # features output: (B, 512, 1, 1, 1)
        feats = self.features(x)
        feats = feats.squeeze(-1).squeeze(-1).squeeze(-1)  # (B, 512)

        logit = self.classifier(feats) # (B, 1)
        return torch.sigmoid(logit).squeeze(1) # (B,)


# Evaluation
def evaluate(model, loader, threshold=None):
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            x, _, y = batch
            x = x.to(DEVICE)
            p = model(x).cpu()
            all_p.extend(p.tolist())
            all_y.extend(y.tolist())

    auroc = roc_auc_score(all_y, all_p)
    prauc = average_precision_score(all_y, all_p)

    if threshold is not None:
        f1    = f1_score(all_y, [1 if p >= threshold else 0 for p in all_p], zero_division=0)
    else:
        f1 = None

    return auroc, prauc, f1, all_p, all_y

def find_best_threshold(all_y, all_p):
    """Find threshold that maximises F1 on validation set."""
    import numpy as np
    thresholds = np.linspace(0.01, 0.99, 200)
    best_f1, best_th = 0.0, 0.5
    for th in thresholds:
        f1 = f1_score(all_y, [1 if p >= th else 0 for p in all_p], zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    return float(best_th), float(best_f1)


# Main 
def main():
    print(f"[R3D-18] Device: {DEVICE}")
    print(f"[R3D-18] Output: {OUT_DIR}")

    # Datasets
    train_ds = DS(data_dir=DATA_DIR, split="TRAIN", augment=True)
    val_ds = DS(data_dir=DATA_DIR, split="VAL",   augment=False)
    test_ds  = DS(data_dir=DATA_DIR, split="TEST",  augment=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"[R3D-18] Train={len(train_ds)} Val={len(val_ds)} "
          f"Test={len(test_ds)}")

    # Model
    model = R3D18Baseline(freeze_backbone=False).to(DEVICE)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.BCELoss()

    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"[R3D-18] Trainable params: {trainable:,}")

    # Training log 
    log_path = os.path.join(OUT_DIR, "train_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_auroc", "val_prauc"])

    best_auroc = 0.0
    patience_count = 0
    best_val_p = None
    best_val_y = None

    # Training loop
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for batch in train_dl:
            x, _, y = batch
            x, y = x.to(DEVICE), y.float().to(DEVICE)

            opt.zero_grad()
            p    = model(x)
            loss = loss_fn(p, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRAD_CLIP)
            opt.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss              = total_loss / n_batches
        val_auroc, val_prauc, _, val_p, val_y = evaluate(model, val_dl)

        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
              f"val_AUROC={val_auroc:.4f} | val_PRAUC={val_prauc:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{avg_loss:.4f}",
                 f"{val_auroc:.4f}", f"{val_prauc:.4f}"])

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience_count = 0
            best_val_p = val_p
            best_val_y = val_y
            torch.save(
                model.state_dict(),
                os.path.join(OUT_DIR, "best_r3d18.pt")
            )
            print(f"  [saved] best model AUROC={best_auroc:.4f} "
                  f"at epoch {epoch}")
        else:
            patience_count += 1
            print(f"  [patience {patience_count}/{PATIENCE}]")
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Threshold calibration on validation set 
    best_threshold, best_val_f1 = find_best_threshold(
        best_val_y, best_val_p)
    print(f"\n[R3D-18] Calibrated threshold: {best_threshold:.4f} "
          f"(val F1={best_val_f1:.4f})")

    # Test evaluation 
    model.load_state_dict(
        torch.load(os.path.join(OUT_DIR, "best_r3d18.pt"), map_location=DEVICE))

    test_auroc, test_prauc, test_f1, test_p, test_y = evaluate(model, test_dl, threshold=best_threshold)

    print(f"\n[R3D-18] Test Results:")
    print(f" AUROC= {test_auroc:.4f}")
    print(f" PR-AUC = {test_prauc:.4f}")
    print(f" F1 = {test_f1:.4f}")
    print(f" Thr = {best_threshold:.4f}")

    # Save predictions
    pred_path = os.path.join(OUT_DIR, "r3d18_test_preds.csv")
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "label", "prob"])
        for i, (y, p) in enumerate(zip(test_y, test_p)):
            writer.writerow([i, y, f"{p:.6f}"])
    print(f"[saved] {pred_path}")

    # Save threshold for plot_results
    import json
    threshold_path = os.path.join(OUT_DIR, "r3d18_threshold.json")
    with open(threshold_path, "w") as f:
        json.dump({"R3D18": best_threshold}, f, indent=2)
    print(f"[saved] {threshold_path}")

    print(f"\n{'='*50}")
    print(f"R3D-18 Baseline — Final Test Results")
    print(f"{'='*50}")
    print(f"AUROC: {test_auroc:.3f}")
    print(f"PR-AUC : {test_prauc:.3f}")
    print(f"F1: {test_f1:.3f}")
    print(f"Thr: {best_threshold:.3f}")
    print(f"{'='*50}")
    print(f"\nAdd to MODELS in plot_results_fixed.py:")
    print(f' "R3D18": "{pred_path}"')
    print(f"\nAdd to config_thresholds.json:")
    print(f' "R3D18": {best_threshold:.4f}')


if __name__ == "__main__":
    main()