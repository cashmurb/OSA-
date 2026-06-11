import os, argparse, json, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.osa import build_model
from training.losses import OSALoss
from evaluation.metrics import compute_metrics
from configs.config import FINETUNE, CLIP


def get_loaders(dataset: str, data_dir: str, batch_size: int, num_workers: int):
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    if dataset == "echonet":
        from datasets.dynamic_dataset import EchoNetDynamicDataset as DS
        tr = DataLoader(DS(data_dir, "TRAIN", augment=True),  shuffle=True,  **kw)
        va = DataLoader(DS(data_dir, "VAL",   augment=False), shuffle=False, **kw)
        te = DataLoader(DS(data_dir, "TEST",  augment=False), shuffle=False, **kw)
    elif dataset == "pediatric":
        from datasets.pediatric_dataset import EchoNetPediatricDataset as DS
        tr = DataLoader(DS(data_dir, "TRAIN", augment=True),  shuffle=True,  **kw)
        va = DataLoader(DS(data_dir, "VAL",   augment=False), shuffle=False, **kw)
        te = DataLoader(DS(data_dir, "TEST",  augment=False), shuffle=False, **kw)
    elif dataset == "camus":
        from datasets.camus_dataset import CAMUSDataset as DS
        tr = DataLoader(DS(data_dir, split="train", augment=True),  shuffle=True,  **kw)
        va = DataLoader(DS(data_dir, split="val",   augment=False), shuffle=False, **kw)
        te = None
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return tr, va, te


def run_epoch(model, loader, criterion, optimiser, scaler, device, use_amp, train):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for step, (video, roi, label) in enumerate(loader):
            video, roi, label = video.to(device), roi.to(device), label.to(device)

            if torch.isnan(video).any():
                print(f"    [WARN] NaN in input at step {step}, skipping.")
                continue

            if train and optimiser:
                optimiser.zero_grad()

            try:
                with autocast(enabled=use_amp):
                    out = model(video, roi)
                    losses = criterion(
                        clip_prob=out["clip_prob"],
                        labels=label,
                        frame_scores=out["frame_scores"],
                        attn_maps=out["oaa_weights"],
                        roi=roi,
                    )

                if torch.isnan(losses["total"]):
                    print(f"    [WARN] NaN loss at step {step}, skipping.")
                    if train and optimiser:
                        optimiser.zero_grad()
                        with torch.no_grad():
                            for p in model.parameters():
                                if p.grad is not None:
                                    p.grad.zero_()
                                if torch.isnan(p).any():
                                    nn.init.normal_(p, mean=0.0, std=0.02)
                    continue

                if train and optimiser:
                    if use_amp:
                        scaler.scale(losses["total"]).backward()
                        scaler.unscale_(optimiser)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), FINETUNE["grad_clip"])
                        scaler.step(optimiser)
                        scaler.update()
                    else:
                        losses["total"].backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), FINETUNE["grad_clip"])
                        optimiser.step()

            except RuntimeError as e:
                print(f"    [WARN] RuntimeError at step {step}: {e}, skipping.")
                if train and optimiser:
                    optimiser.zero_grad()
                continue

            total_loss += losses["total"].item()
            probs_batch = torch.sigmoid(out["clip_prob"]).detach().cpu()
            if not torch.isnan(probs_batch).any():
                all_probs.append(probs_batch)
                all_labels.append(label.detach().cpu())

            if train and step % 20 == 0:
                print(
                    f"    step {step:4d}/{len(loader)} | "
                    f"loss={losses['total'].item():.4f}  "
                    f"cls={losses['cls'].item():.4f}  "
                    f"reg={losses['reg'].item():.4f}  "
                    f"smooth={losses['smooth'].item():.4f}"
                )

    if len(all_probs) == 0:
        print("  [WARN] No valid batches this epoch.")
        return {"loss": float("nan"), "auroc": 0.0, "f1": 0.0, "pr_auc": 0.0,
                "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = np.nan_to_num(probs, nan=0.5)
    metrics = compute_metrics(labels, probs)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(FINETUNE["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # AMP disabled for M3 to avoid fp16 overflow with pretrained TAH weights
    use_amp = FINETUNE["mixed_precision"] and device.type == "cuda" and args.variant not in ("M3",)
    print(f"[{args.variant}] Device: {device}  AMP: {use_amp}")

    tr_loader, va_loader, te_loader = get_loaders(
        args.dataset, args.data_dir,
        FINETUNE["batch_size"], FINETUNE["num_workers"],
    )
    print(f"[{args.variant}] Train: {len(tr_loader.dataset)}  Val: {len(va_loader.dataset)}")

    model = build_model(
        args.variant,
        pretrained=True,
        freeze_core=True,
        tah_pretrain_ckpt=getattr(args, "tah_pretrain_ckpt", None),
    ).to(device)

    if args.ssl_backbone and os.path.isfile(args.ssl_backbone):
        state = torch.load(args.ssl_backbone, map_location=device)
        missing, unexpected = model.backbone.load_state_dict(state, strict=False)
        print(f"[{args.variant}] Loaded SSL backbone. "
              f"Missing: {len(missing)}  Unexpected: {len(unexpected)}")

    # M4/M5 use smoothing; M2/M3 use cls loss only
    lambda_smooth = FINETUNE.get("lambda_smooth_spatial", 0.1) if args.variant in ("M4", "M5") else 0.0
    criterion = OSALoss(lambda_smooth=lambda_smooth)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = AdamW(trainable, lr=args.lr, weight_decay=FINETUNE["weight_decay"])
    scheduler = CosineAnnealingLR(
        optimiser,
        T_max=args.epochs - FINETUNE["warmup_epochs"],
        eta_min=1e-6,
    )
    scaler = GradScaler(enabled=use_amp)

    os.makedirs(args.output, exist_ok=True)

    start_epoch = 1
    best_auroc = 0.0
    patience_ctr = 0
    history = {"variant": args.variant, "train": [], "val": []}

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        start_epoch = ckpt["epoch"] + 1
        best_auroc = ckpt.get("best_auroc", 0.0)
        history = ckpt.get("history", history)
        print(f"[{args.variant}] Resumed from epoch {ckpt['epoch']}")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"\n── {args.variant} Epoch {epoch}/{args.epochs} ──")

        tr = run_epoch(model, tr_loader, criterion, optimiser, scaler, device, use_amp, train=True)
        va = run_epoch(model, va_loader, criterion, None, None, device, use_amp, train=False)

        if epoch > FINETUNE["warmup_epochs"]:
            scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Train: loss={tr['loss']:.4f}  AUROC={tr.get('auroc',0):.4f}\n"
            f"  Val:   loss={va['loss']:.4f}  AUROC={va.get('auroc',0):.4f}  "
            f"F1={va.get('f1',0):.4f}  [{elapsed:.0f}s]"
        )

        history["train"].append(tr)
        history["val"].append(va)

        val_auroc = va.get("auroc", 0.0)
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience_ctr = 0
            best_path = os.path.join(args.output, f"best_{args.variant.lower()}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "best_auroc": best_auroc,
                "history": history,
            }, best_path)
            print(f"  ✔ Best checkpoint (AUROC={best_auroc:.4f}): {best_path}")
        else:
            patience_ctr += 1
            if patience_ctr >= FINETUNE["early_stop_patience"]:
                print(f"  Early stopping at epoch {epoch}.")
                break

        if epoch % FINETUNE["save_every"] == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "best_auroc": best_auroc,
                "history": history,
            }, os.path.join(args.output, f"ckpt_epoch_{epoch:03d}.pt"))

    if te_loader is not None:
        best_path = os.path.join(args.output, f"best_{args.variant.lower()}.pt")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        te = run_epoch(model, te_loader, criterion, None, None, device, use_amp, train=False)
        print(f"\n[{args.variant}] Test: {te}")
        history["test"] = te

    result_path = os.path.join(args.output, f"{args.variant.lower()}_results.json")
    with open(result_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[{args.variant}] Results → {result_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="M1",
                   choices=["M1", "M1_DEV", "M2", "M3", "M3_DEV", "M4", "M5"])
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dataset", default="pediatric",
                   choices=["echonet", "pediatric", "camus"])
    p.add_argument("--epochs", type=int,   default=FINETUNE["epochs"])
    p.add_argument("--lr", type=float, default=FINETUNE["lr"])
    p.add_argument("--ssl_backbone", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--tah_pretrain_ckpt", default=None,
                   help="Path to tah_pretrain_best.pt — required for M3")
    args = p.parse_args()
    if args.output is None:
        args.output = f"experiments/{args.variant.lower()}"
    main(args)
