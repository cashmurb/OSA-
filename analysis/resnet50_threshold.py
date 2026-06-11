import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/echonet-pediatric"
CKPT = "experiments/baseline_resnet50/best_resnet50.pt"

from datasets.pediatric_dataset import EchoNetPediatricDataset as DS

class ResNet50Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=None)
        self.features   = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(2048, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x     = x.view(B * T, C, H, W)
        feats = self.features(x).squeeze(-1).squeeze(-1)
        feats = feats.view(B, T, 2048).mean(dim=1)
        return torch.sigmoid(self.classifier(feats)).squeeze(1)

def main():
    model = ResNet50Baseline().to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    if isinstance(state, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in state:
                state = state[key]
                break
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded: {CKPT}")

    # Val set
    val_ds = DS(data_dir=DATA_DIR, split="VAL", augment=False)
    val_dl = DataLoader(val_ds, batch_size=8,
                        shuffle=False, num_workers=4)

    all_p, all_y = [], []
    with torch.no_grad():
        for batch in val_dl:
            x, _, y = batch
            x = x.to(DEVICE)
            p = model(x).cpu()
            all_p.extend(p.tolist())
            all_y.extend(y.tolist())

    # Calibrate threshold on val set
    thresholds = np.linspace(0.01, 0.99, 500)
    best_f1, best_th = 0.0, 0.5
    for th in thresholds:
        f1 = f1_score(all_y,
                      [1 if p >= th else 0 for p in all_p],
                      zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th

    val_auroc = roc_auc_score(all_y, all_p)
    val_prauc = average_precision_score(all_y, all_p)

    print(f"\nResNet-50 Validation Results:")
    print(f" AUROC  = {val_auroc:.4f}")
    print(f" PR-AUC = {val_prauc:.4f}")
    print(f" Best threshold = {best_th:.4f}")
    print(f" Val F1 at threshold = {best_f1:.4f}")

    # Now evaluate test set with this threshold
    test_ds = DS(data_dir=DATA_DIR, split="TEST", augment=False)
    test_dl = DataLoader(test_ds, batch_size=8,
                         shuffle=False, num_workers=4)

    test_p, test_y = [], []
    with torch.no_grad():
        for batch in test_dl:
            x, _, y = batch
            x = x.to(DEVICE)
            p = model(x).cpu()
            test_p.extend(p.tolist())
            test_y.extend(y.tolist())

    test_auroc = roc_auc_score(test_y, test_p)
    test_prauc = average_precision_score(test_y, test_p)
    test_f1    = f1_score(test_y,
                          [1 if p >= best_th else 0 for p in test_p],
                          zero_division=0)
    from sklearn.metrics import precision_score, recall_score
    test_prec  = precision_score(test_y,
                                 [1 if p >= best_th else 0
                                  for p in test_p], zero_division=0)
    test_rec   = recall_score(test_y,
                              [1 if p >= best_th else 0
                               for p in test_p], zero_division=0)

    print(f"\nResNet-50 Test Results (threshold={best_th:.4f}):")
    print(f" AUROC = {test_auroc:.4f}")
    print(f" PR-AUC = {test_prauc:.4f}")
    print(f" F1 = {test_f1:.4f}")
    print(f" Precision = {test_prec:.4f}")
    print(f" Recall = {test_rec:.4f}")
    print(f" Threshold = {best_th:.4f}")

    # Save threshold to json
    import json
    out = {
        "ResNet50": best_th,
        "ResNet50_val_f1": best_f1,
        "ResNet50_test_auroc": test_auroc,
        "ResNet50_test_prauc": test_prauc,
        "ResNet50_test_f1": test_f1,
        "ResNet50_test_precision": test_prec,
        "ResNet50_test_recall": test_rec,
    }
    with open("experiments/baseline_resnet50/resnet50_final_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] resnet50_final_results.json")

    # Add to config_thresholds.json
    cfg_path = "analysis/config_thresholds.json"
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["ResNet50"] = best_th
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[updated] config_thresholds.json with ResNet50={best_th:.4f}")

if __name__ == "__main__":
    main()