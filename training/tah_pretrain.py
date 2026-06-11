import os
import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.backbone_uniformer import UniFormerV2Backbone
from models.temporal_head import TemporalAnomalyHead
from configs.config import BACKBONE, TAH, CLIP

try:
    from datasets.pediatric_dataset import EchoNetPediatricDataset as DS
except ImportError:
    DS = None

def mtr_loss(V_orig, V_hat, mask):
    """MSE at masked positions only."""
    return nn.functional.mse_loss(V_hat[mask], V_orig[mask].detach())

def run_pretrain(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MTR] Running on {device}")

    backbone = UniFormerV2Backbone(
        pretrained=args.pretrained_backbone,
        freeze_core=True,
    ).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    tah = TemporalAnomalyHead(in_dim=backbone.feature_dim).to(device)

    # frame_scorer excluded — don't let reconstruction objective bias anomaly scoring
    pretrain_params = (
        list(tah.proj.parameters())
        + [tah.mask_token, tah.pos_embed]
        + list(tah.encoder.parameters())
        + list(tah.recon_head.parameters())
    )

    optimizer = AdamW(pretrain_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    if DS is None:
        raise ImportError("Could not import EchoNetPediatricDataset.")

    # combine train + val (no labels needed for MTR)
    full_ds = ConcatDataset([
        DS(data_dir=args.data_dir, split="train"),
        DS(data_dir=args.data_dir, split="val"),
    ])
    loader = DataLoader(
        full_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    print(f"[MTR] {len(full_ds)} clips | {len(loader)} batches/epoch")

    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = math.inf

    for epoch in range(1, args.epochs + 1):
        tah.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            video = batch[0] if isinstance(batch, (list, tuple)) else batch
            video = video.to(device, non_blocking=True)

            with torch.no_grad():
                feats = backbone(video)
                B, T, Hp, Wp, C = feats.shape
                flat = feats.reshape(B, T, Hp * Wp, C)

            V_orig, V_hat, mask = tah.forward_mtr(flat, mask_ratio=args.mask_ratio)
            loss = mtr_loss(V_orig, V_hat, mask)

            if torch.isnan(loss):
                print(f"[MTR] WARNING: NaN loss at epoch {epoch}, skipping batch.")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pretrain_params, max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"[MTR] Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(args.save_dir, "tah_pretrain_best.pt")
            torch.save(tah.state_dict(), ckpt_path)
            print(f"[MTR]   → Saved best checkpoint (loss={best_loss:.6f})")

        if epoch % args.save_every == 0:
            torch.save(
                tah.state_dict(),
                os.path.join(args.save_dir, f"tah_pretrain_ep{epoch:03d}.pt"),
            )

    print(f"[MTR] Done. Best loss: {best_loss:.6f}")
    print(f"[MTR] Weights: {args.save_dir}/tah_pretrain_best.pt")

def parse_args():
    p = argparse.ArgumentParser(description="TAH Masked Temporal Reconstruction Pre-training")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int,  default=8)
    p.add_argument("--mask_ratio", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--pretrained_backbone", action="store_true", default=True)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pretrain(args)
