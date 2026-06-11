import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import glob
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from datasets.pediatric_dataset import EchoNetPediatricDataset as PediatricDataset
from models.osa import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/echonet-pediatric"
OUT_DIR = "analysis/figures/gradcam"
os.makedirs(OUT_DIR, exist_ok=True)

# USER CONFIGURATION (adjust these for each run)
VARIANT = "M5" # one of "M1", "M2", "M4", "M5"
EXP_DIR = "experiments/finetune_M4R_v1"
PRED_CSV = os.path.join(EXP_DIR, "m5_test_preds.csv")

# thresholds for automatic case selection (from validation optimisation)
THRESHOLDS = {
    "M1": 0.135,
    "M2": 0.158,
    "M3": 0.055, 
    "M4": 0.085,
    "M5": 0.087,
}
THRESHOLD = THRESHOLDS[VARIANT]

# Optional: force a single case instead of automatic TP/TN/FP/FN selection
FORCE_CASE = None      # e.g. "FP" or None
FORCE_INDEX = None     # e.g. 575 or None

def find_checkpoint(expdir: str, variant: str) -> str:
    candidates = [
        os.path.join(expdir, f"best_{variant.lower()}.pt"),
        os.path.join(expdir, "best.pt"),
        os.path.join(expdir, f"{variant.lower()}_best.pt"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    extra = sorted(glob.glob(os.path.join(expdir, "*.pt")))
    if extra:
        return extra[0]
    raise FileNotFoundError(f"No checkpoint found in {expdir}")

def load_model():
    ckpt_path = find_checkpoint(EXP_DIR, VARIANT)
    model = build_model(VARIANT, pretrained=True, freeze_core=True).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"[{VARIANT}] Loaded checkpoint: {ckpt_path}")
    return model

def load_rows(path: str):
    rows = []
    with open(path) as fp:
        r = csv.DictReader(fp)
        for row in r:
            rows.append({
                "idx": int(row["idx"]),
                "label": int(float(row["label"])),
                "prob": float(row["prob"]),
                "path": row["path"],
            })
    return rows

def load_rows_by_idx(path: str):
    rows_by_idx = {}
    with open(path) as fp:
        r = csv.DictReader(fp)
        for row in r:
            idx = int(row["idx"])
            rows_by_idx[idx] = {
                "idx": idx,
                "label": int(float(row["label"])),
                "prob": float(row["prob"]),
                "path": row["path"],
            }
    return rows_by_idx

def select_cases(rows, threshold: float):
    tp = [r for r in rows if r["label"] == 1 and r["prob"] >= threshold]
    tn = [r for r in rows if r["label"] == 0 and r["prob"] < threshold]
    fp = [r for r in rows if r["label"] == 0 and r["prob"] >= threshold]
    fn = [r for r in rows if r["label"] == 1 and r["prob"] < threshold]

    tp = sorted(tp, key=lambda x: -x["prob"])
    tn = sorted(tn, key=lambda x: x["prob"])
    fp = sorted(fp, key=lambda x: -x["prob"])
    fn = sorted(fn, key=lambda x: x["prob"])

    chosen = {}
    if tp: chosen["TP"] = tp[0]
    if tn: chosen["TN"] = tn[0]
    if fp: chosen["FP"] = fp[0]
    if fn: chosen["FN"] = fn[0]
    return chosen

def denorm_frame(frame_tensor):
    img = frame_tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = (img + 1.0) / 2.0
    return np.clip(img, 0, 1)

def overlay_heatmap(img, cam, alpha=0.4):
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    cmap = plt.get_cmap("jet")
    heat = cmap(cam)[..., :3]
    over = (1 - alpha) * img + alpha * heat
    return np.clip(over, 0, 1)

def extract_backbone_feats(model, video):
    return model.backbone(video)

def forward_from_feats(model, feats, roi):
    """
    Reconstruct the per‑variant forward path from backbone features.
    """
    if VARIANT == "M1":
        B, T, H, W, C = feats.shape
        flat = feats.reshape(B, T, H * W, C)
        cls_tokens = flat.mean(dim=2)
        clip_logit = model.classifier(cls_tokens=cls_tokens)
        return clip_logit

    elif VARIANT in ("M2", "M3"):
        B, T, H, W, C = feats.shape
        flat = feats.reshape(B, T, H * W, C)
        Ht, frame_scores = model.tah(flat)
        clip_logit = model.classifier(H=Ht)
        return clip_logit

    elif VARIANT in ("M4", "M5"):
        with torch.autocast(device_type="cuda", enabled=False):
            F_roi, _ = model.oaa_r(feats.float(), roi.float() if roi is not None else None)
        F_roi = F_roi.to(feats.dtype)
        Ht, frame_scores = model.tah(F_roi)
        smoothed = model.smoother(frame_scores)
        clip_logit = smoothed.mean(dim=1)
        return clip_logit

    else:
        raise ValueError(f"Unsupported VARIANT for Grad-CAM: {VARIANT}")

def compute_gradcam(model, video, roi):
    """
    video: (1, T, C, H, W)
    returns:
        cams_per_t: list of (H, W) CAMs, length T_backbone
        clip_prob
    """
    feats = extract_backbone_feats(model, video)
    feats = feats.detach().requires_grad_(True)

    clip_logit = forward_from_feats(model, feats, roi)
    clip_prob = torch.sigmoid(clip_logit).item()

    model.zero_grad(set_to_none=True)
    clip_logit.backward(torch.ones_like(clip_logit))

    grads = feats.grad
    acts = feats

    cams = []
    B, T, Hp, Wp, C = acts.shape
    for t in range(T):
        g = grads[0, t] # (H', W', C)
        a = acts[0, t]  # (H', W', C)

        weights = g.mean(dim=(0, 1)) # (C,)
        cam = (a * weights).sum(dim=-1) # (H', W')
        cam = F.relu(cam)

        cam = cam.unsqueeze(0).unsqueeze(0)  # (1,1,H',W')
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cams.append(cam)

    return cams, clip_prob

def save_case(model, ds, row, case_name):
    idx = row["idx"]
    video, roi, label = ds[idx]
    video_b = video.unsqueeze(0).to(DEVICE)
    roi_b = roi.unsqueeze(0).to(DEVICE)

    cams, prob = compute_gradcam(model, video_b, roi_b)

    out_dir = os.path.join(OUT_DIR, VARIANT.lower(), f"{case_name.lower()}_{idx}")
    os.makedirs(out_dir, exist_ok=True)

    # Map 8 backbone temporal tokens to 16 video frames
    T_backbone = len(cams)
    T_video = video.shape[0]
    mapping = np.linspace(0, T_video - 1, T_backbone).astype(int)

    for t, frame_idx in enumerate(mapping):
        img = denorm_frame(video[frame_idx])
        over = overlay_heatmap(img, cams[t], alpha=0.4)

        plt.figure(figsize=(4, 4))
        plt.imshow(over)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cam_t{t}_f{frame_idx}.png"),
                    dpi=200, bbox_inches="tight", pad_inches=0)
        plt.close()

    # Average CAM
    cam_mean = np.mean(np.stack(cams, axis=0), axis=0)
    mid_frame = video[mapping[len(mapping)//2]]
    img_mid = denorm_frame(mid_frame)
    over_mean = overlay_heatmap(img_mid, cam_mean, alpha=0.4)

    plt.figure(figsize=(4, 4))
    plt.imshow(over_mean)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cam_mean.png"),
                dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"{case_name}: idx={idx} label={label.item():.0f} prob={prob:.4f} saved to {out_dir}")

    # Save raw frame for comparison
    img_raw = denorm_frame(video[mapping[len(mapping)//2]])
    plt.figure(figsize=(4, 4))
    plt.imshow(img_raw, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "raw_frame.png"),
                dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

def main():
    ds = PediatricDataset(data_dir=DATA_DIR, split="TEST")
    model = load_model()

    if FORCE_CASE is not None and FORCE_INDEX is not None:
        # Forced single case
        rows_by_idx = load_rows_by_idx(PRED_CSV)
        if FORCE_INDEX not in rows_by_idx:
            raise ValueError(f"Index {FORCE_INDEX} not found in {PRED_CSV}")

        row = rows_by_idx[FORCE_INDEX]
        pred = 1 if row["prob"] >= THRESHOLD else 0

        print(f"[{VARIANT}] threshold={THRESHOLD}")
        print(f"  FORCED {FORCE_CASE}: idx={row['idx']} label={row['label']} prob={row['prob']:.4f} "
              f"pred={pred} path={row['path']}")

        save_case(model, ds, row, FORCE_CASE)

    else:
        # Automatic selection of TP, TN, FP, FN 
        rows = load_rows(PRED_CSV)
        chosen = select_cases(rows, THRESHOLD)

        print(f"[{VARIANT}] threshold={THRESHOLD}")
        for k, v in chosen.items():
            print(f"  {k}: idx={v['idx']} label={v['label']} prob={v['prob']:.4f} path={v['path']}")

        for case_name, row in chosen.items():
            save_case(model, ds, row, case_name)

if __name__ == "__main__":
    main()