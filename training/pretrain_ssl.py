# Phase 1 Self-Supervised Pretraining (MFM + TOP)

import os, sys, argparse, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.backbone_uniformer import UniFormerV2Backbone
from training.losses import SSLLoss
from configs.config import SSL, CLIP

# SSL heads
class MFMHead(nn.Module):
    """Reconstructs masked spatiotemporal patch features."""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, N, C = feats.shape
        flat  = feats.reshape(B * T * N, C)
        m_flat = mask.reshape(B * T * N)
        return self.proj(flat[m_flat])

class TOPHead(nn.Module):
    """Predicts whether a frame sequence is in temporal order (binary)."""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, T, N, C') → logit (B,)"""
        x = feats.mean(dim=(1, 2))   # global pool → (B, C')
        return self.fc(x).squeeze(-1)

# Helpers
def make_mask(B: int, T: int, N: int, ratio: float, device) -> torch.Tensor:
    """Random binary mask (B, T, N) — True = masked patch."""
    total = T * N
    n_mask = int(total * ratio)
    mask = torch.zeros(B, total, dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(total, device=device)[:n_mask]
        mask[b, idx] = True
    return mask.view(B, T, N)

def shuffle_order(video: torch.Tensor) -> tuple:
    """Randomly shuffle ~50% of clips; return (shuffled, labels)."""
    B, T = video.shape[:2]
    out = video.clone()
    labels = torch.zeros(B, device=video.device)
    for b in range(B):
        if random.random() > 0.5:
            perm = torch.randperm(T, device=video.device)
            out[b] = video[b, perm]
            labels[b] = 1.0
    return out, labels

def get_dataset(name: str, data_dir: str, augment: bool = False):
    if name == "echonet":
        # EchoNet-Dynamic: large adult dataset — primary SSL source
        from datasets.dynamic_dataset import EchoNetDynamicDataset as DS
        return ConcatDataset([
            DS(data_dir, split="TRAIN", augment=augment),
            DS(data_dir, split="VAL",   augment=False),
        ])
    elif name == "pediatric":
        # EchoNet-Pediatric: use unlabelled splits for SSL
        from datasets.pediatric_dataset import EchoNetPediatricDataset as DS
        return ConcatDataset([
            DS(data_dir, split="TRAIN", augment=augment),
            DS(data_dir, split="VAL",   augment=False),
        ])
    elif name == "camus":
        from datasets.camus_dataset import CAMUSDataset as DS
        return DS(data_dir, split="train", augment=augment)
    raise ValueError(f"Unknown dataset: {name}")


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# Main
def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[SSL] Device: {device}  AMP: {use_amp}")

    # Dataset 
    ds = get_dataset(args.dataset, args.data_dir, augment=False)
    loader = DataLoader(
        ds, batch_size=SSL["batch_size"], shuffle=True,
        num_workers=SSL["num_workers"], pin_memory=True, drop_last=True,
    )
    print(f"[SSL] Samples: {len(ds)}  Batches/epoch: {len(loader)}")

    # Model 
    backbone = UniFormerV2Backbone(
        pretrained=True,
        freeze_core=False, # full backbone is trained during SSL
    ).to(device)

    # Probe feature shape
    with torch.no_grad():
        dummy = torch.zeros(1, CLIP["num_frames"], CLIP["channels"],
                            CLIP["img_size"], CLIP["img_size"], device=device)
        f = backbone(dummy)           # (1, T, H', W', C')
    _, T, Hp, Wp, C = f.shape
    N = Hp * Wp
    print(f"[SSL] Feature shape: T={T}, H'={Hp}, W'={Wp}, C'={C}, N={N}")

    mfm_head = MFMHead(C).to(device)
    top_head = TOPHead(C).to(device)
    criterion = SSLLoss(beta=args.beta)

    optimiser = AdamW(
        list(backbone.parameters()) +
        list(mfm_head.parameters()) +
        list(top_head.parameters()),
        lr=args.lr, weight_decay=SSL["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimiser, T_max=args.epochs - SSL["warmup_epochs"], eta_min=1e-6
    )
    scaler = GradScaler(enabled=use_amp)

    os.makedirs(args.output, exist_ok=True)

    # Resume 
    start_epoch = 1
    history = []
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        backbone.load_state_dict(ckpt["backbone"])
        mfm_head.load_state_dict(ckpt["mfm_head"])
        top_head.load_state_dict(ckpt["top_head"])
        optimiser.load_state_dict(ckpt["optimiser"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        print(f"[SSL] Resumed from epoch {ckpt['epoch']}")

    # Training loop 
    for epoch in range(start_epoch, args.epochs + 1):
        backbone.train(); mfm_head.train(); top_head.train()
        ep_mfm = ep_top = ep_total = 0.0
        t0 = time.time()

        for video, _, _ in loader:
            video = video.to(device) # (B, T, C, H, W)

            # TOP: shuffle some sequences
            video_top, top_labels = shuffle_order(video)

            optimiser.zero_grad()
            with autocast(enabled=use_amp):
                # Features
                feats = backbone(video) # (B, T, H', W', C')
                feats_top = backbone(video_top)

                # Flatten spatial → (B, T, N, C')
                B2, T2, Hp2, Wp2, C2 = feats.shape
                N2 = Hp2 * Wp2
                feats_flat = feats.reshape(B2, T2, N2, C2)
                feats_top_flat = feats_top.reshape(B2, T2, N2, C2)

                # MFM
                mask = make_mask(B2, T2, N2, SSL["mask_ratio"], device)
                recon = mfm_head(feats_flat, mask)
                target = feats_flat.reshape(B2 * T2 * N2, C2)[mask.reshape(-1)].detach()

                # TOP
                top_logits = top_head(feats_top_flat)

                losses = criterion(recon, target, top_logits, top_labels)

            if use_amp:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 0.5)
                scaler.step(optimiser)
                scaler.update()
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 0.5)
                optimiser.step()

            ep_mfm += losses["mfm"].item()
            ep_top += losses["top"].item()
            ep_total += losses["total"].item()

        # Warmup then cosine schedule
        if epoch > SSL["warmup_epochs"]:
            scheduler.step()

        n = len(loader)
        row = {
            "epoch": epoch,
            "mfm": ep_mfm   / n,
            "top": ep_top   / n,
            "total": ep_total / n,
            "time_s": round(time.time() - t0, 1),
        }
        history.append(row)
        print(
            f"SSL {epoch:3d}/{args.epochs} | "
            f"MFM={row['mfm']:.4f}  TOP={row['top']:.4f}  "
            f"Total={row['total']:.4f}  [{row['time_s']}s]"
        )

        # Checkpoint 
        if epoch % SSL["save_every"] == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.output, f"ckpt_epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "backbone": backbone.state_dict(),
                "mfm_head": mfm_head.state_dict(),
                "top_head": top_head.state_dict(),
                "optimiser": optimiser.state_dict(),
                "history": history,
            }, ckpt_path)
            # Keep a symlink to latest checkpoint for easy resuming
            latest = os.path.join(args.output, "ckpt_latest.pt")
            if os.path.islink(latest):
                os.remove(latest)
            os.symlink(os.path.abspath(ckpt_path), latest)
            print(f" ✔ Checkpoint: {ckpt_path}")

    # Save final backbone
    final_path = os.path.join(args.output, "ssl_backbone_final.pt")
    torch.save(backbone.state_dict(), final_path)
    with open(os.path.join(args.output, "ssl_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[SSL] Done. Backbone saved: {final_path}")
    print(f"[SSL] To resume if interrupted: --resume {args.output}/ckpt_latest.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dataset", default="echonet", choices=["echonet","pediatric","camus"])
    p.add_argument("--epochs", type=int,   default=SSL["epochs"])
    p.add_argument("--lr", type=float, default=SSL["lr"])
    p.add_argument("--beta", type=float, default=SSL["beta"])
    p.add_argument("--output", default="experiments/ssl")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    main(p.parse_args())