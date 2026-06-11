# training/train_baseline.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import roc_auc_score, average_precision_score
from datasets.pediatric_dataset import EchoNetPediatricDataset as DS

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR  = "experiments/baseline_resnet50"
DATA_DIR = "data/echonet-pediatric"
os.makedirs(OUT_DIR, exist_ok=True)

LR         = 3e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8
MAX_EPOCHS = 50
PATIENCE   = 10

class ResNet50Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(2048, 1)
        # Initialise bias conservatively like M0
        nn.init.constant_(self.classifier.bias, -2.0)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.features(x).squeeze(-1).squeeze(-1)  # (B*T, 2048)
        feats = feats.view(B, T, 2048).mean(dim=1) # (B, 2048)
        logit = self.classifier(feats)  # (B, 1)
        return torch.sigmoid(logit).squeeze(1) # (B,)

def evaluate(model, loader):
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            x, _, y = batch
            x = x.to(DEVICE)
            p = model(x).cpu()
            all_p.extend(p.tolist())
            all_y.extend(y.tolist())
    auroc  = roc_auc_score(all_y, all_p)
    prauc  = average_precision_score(all_y, all_p)
    return auroc, prauc, all_p, all_y

def main():
    train_ds = DS(data_dir=DATA_DIR, split="TRAIN", augment=True)
    val_ds = DS(data_dir=DATA_DIR, split="VAL",   augment=False)
    test_ds = DS(data_dir=DATA_DIR, split="TEST",  augment=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = ResNet50Baseline().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCELoss()

    best_auroc = 0.0
    patience_count = 0
    log_path = os.path.join(OUT_DIR, "train_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_auroc", "val_prauc"])

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            x, _, y = batch
            x, y = x.to(DEVICE), y.float().to(DEVICE)
            opt.zero_grad()
            p = model(x)
            loss = loss_fn(p, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)
        val_auroc, val_prauc, _, _ = evaluate(model, val_dl)

        print(f"Epoch {epoch:3d} | loss={avg_loss:.4f} | "
              f"val_AUROC={val_auroc:.4f} | val_PRAUC={val_prauc:.4f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{avg_loss:.4f}",
                 f"{val_auroc:.4f}", f"{val_prauc:.4f}"])

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience_count = 0
            torch.save(model.state_dict(),
                       os.path.join(OUT_DIR, "best_resnet50.pt"))
            print(f"  [saved] best model at epoch {epoch}")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test evaluation
    model.load_state_dict(
        torch.load(os.path.join(OUT_DIR, "best_resnet50.pt"),
                   map_location=DEVICE))
    test_auroc, test_prauc, test_p, test_y = evaluate(model, test_dl)
    print(f"\nTest AUROC={test_auroc:.4f} | Test PRAUC={test_prauc:.4f}")

    pred_path = os.path.join(OUT_DIR, "resnet50_test_preds.csv")
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "label", "prob"])
        for i, (y, p) in enumerate(zip(test_y, test_p)):
            writer.writerow([i, y, f"{p:.6f}"])
    print(f"[saved] {pred_path}")

if __name__ == "__main__":
    main()