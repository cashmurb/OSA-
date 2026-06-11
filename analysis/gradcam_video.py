import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import glob
from typing import Dict, List, Optional, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from datasets.pediatric_dataset import EchoNetPediatricDataset as PediatricDataset
from models.osa import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data/echonet-pediatric"
OUT_DIR = "analysis/figures/gradcam_videos"
os.makedirs(OUT_DIR, exist_ok=True)

# VARIANT CONFIGURATION
VARIANT_CONFIGS: Dict[str, Dict[str, str | float]] = {
    "M1": {
        "exp_dir": "experiments/finetune_M1_v4",
        "pred_csv": "experiments/finetune_M1_v4/m1_test_preds.csv",
        "threshold": 0.135,
    },
    "M2": {
        "exp_dir": "experiments/finetune_M2_v4",
        "pred_csv": "experiments/finetune_M2_v4/m2_test_preds.csv",
        "threshold": 0.158,
    },
    "M3": {
        "exp_dir": "experiments/finetune_M3_v2",
        "pred_csv": "experiments/finetune_M3_v2/m3_test_preds.csv",
        "threshold": 0.055,
    },
    "M4": {
        "exp_dir": "experiments/finetune_M4_v1",
        "pred_csv": "experiments/finetune_M4_v1/m4_test_preds.csv",
        "threshold": 0.085,
    },
    "M5": {
        "exp_dir": "experiments/finetune_M5_v1",
        "pred_csv": "experiments/finetune_M5_v1/m5_test_preds.csv",
        "threshold": 0.087,
    },
}

# Which variants to render
VARIANTS_TO_RUN = ["M1", "M2", "M3", "M4", "M5"]

# Use one reference variant to choose shared cases
CASE_SELECTION_VARIANT = "M2"

# If None, auto-select 8 TP, 8 TN, 8 FP, 8 FN from CASE_SELECTION_VARIANT
# If not None, use these exact indices for all variants.
FORCED_CASES = [("TP", 379), ("TN", 633), ("FP", 479), ("FN", 136)]
# Example:
# FORCED_CASES = [
#     ("TP1", 379), ("TN1", 633), ("FP1", 62), ("FN1", 136),
#     ("TP2", 102), ("TN2", 210), ("FP2", 455), ("FN2", 501),
# ]

FPS = 8
GIF_FPS = 6
ALPHA = 0.45
FRAME_HW = (224, 224)
SAVE_GIF = False

# 8 TP + 8 TN + 8 FP + 8 FN = 32 total cases
N_PER_CLASS = 8

# If True: strongest-confidence examples only
# If False: spread across easy + moderate cases
USE_EXTREME_CASES = False

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

def load_model_for_variant(variant: str, exp_dir: str) -> torch.nn.Module:
    ckpt_path = find_checkpoint(exp_dir, variant)
    model = build_model(variant, pretrained=True, freeze_core=True).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(ckpt, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"[{variant}] Loaded checkpoint: {ckpt_path}")
    return model

def load_rows(path: str) -> List[Dict[str, int | float | str]]:
    rows: List[Dict[str, int | float | str]] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "idx": int(row["idx"]),
                "label": int(float(row["label"])),
                "prob": float(row["prob"]),
                "path": row.get("path", ""),
            })
    return rows

def pick_spread(samples: List[Dict[str, int | float | str]], n: int) -> List[Dict[str, int | float | str]]:
    if len(samples) <= n:
        return samples
    idxs = np.linspace(0, len(samples) - 1, n, dtype=int)
    return [samples[i] for i in idxs]

def select_n_per_class(
    rows: List[Dict[str, int | float | str]],
    threshold: float,
    n: int = 8,
    extreme: bool = False,
) -> Dict[str, Dict[str, int | float | str]]:
    tp = sorted(
        [r for r in rows if int(r["label"]) == 1 and float(r["prob"]) >= threshold],
        key=lambda x: -float(x["prob"])
    )
    tn = sorted(
        [r for r in rows if int(r["label"]) == 0 and float(r["prob"]) < threshold],
        key=lambda x: float(x["prob"])
    )
    fp = sorted(
        [r for r in rows if int(r["label"]) == 0 and float(r["prob"]) >= threshold],
        key=lambda x: -float(x["prob"])
    )
    fn = sorted(
        [r for r in rows if int(r["label"]) == 1 and float(r["prob"]) < threshold],
        key=lambda x: float(x["prob"])
    )

    if len(tp) < n or len(tn) < n or len(fp) < n or len(fn) < n:
        raise ValueError(
            f"Not enough cases: TP={len(tp)}, TN={len(tn)}, FP={len(fp)}, FN={len(fn)}, need {n} each."
        )

    if extreme:
        tp = tp[:n]
        tn = tn[:n]
        fp = fp[:n]
        fn = fn[:n]
    else:
        tp = pick_spread(tp, n)
        tn = pick_spread(tn, n)
        fp = pick_spread(fp, n)
        fn = pick_spread(fn, n)

    chosen: Dict[str, Dict[str, int | float | str]] = {}
    for i in range(n):
        chosen[f"TP{i+1}"] = tp[i]
        chosen[f"TN{i+1}"] = tn[i]
        chosen[f"FP{i+1}"] = fp[i]
        chosen[f"FN{i+1}"] = fn[i]

    return chosen

def load_forced_cases(
    pred_csv: str,
    forced_cases: List[Tuple[str, int]],
) -> Dict[str, Dict[str, int | float | str]]:
    rows_by_idx: Dict[int, Dict[str, int | float | str]] = {}
    with open(pred_csv) as f:
        for row in csv.DictReader(f):
            i = int(row["idx"])
            rows_by_idx[i] = {
                "idx": i,
                "label": int(float(row["label"])),
                "prob": float(row["prob"]),
                "path": row.get("path", ""),
            }

    missing = [idx for _, idx in forced_cases if idx not in rows_by_idx]
    if missing:
        raise KeyError(f"These forced indices are missing from {pred_csv}: {missing}")

    return {name: rows_by_idx[idx] for name, idx in forced_cases}


def get_shared_cases() -> Dict[str, Dict[str, int | float | str]]:
    pred_csv = str(VARIANT_CONFIGS[CASE_SELECTION_VARIANT]["pred_csv"])
    if FORCED_CASES is not None:
        return load_forced_cases(pred_csv, FORCED_CASES)

    rows = load_rows(pred_csv)
    threshold = float(VARIANT_CONFIGS[CASE_SELECTION_VARIANT]["threshold"])
    return select_n_per_class(rows, threshold, n=N_PER_CLASS, extreme=USE_EXTREME_CASES)


def denorm_frame(frame_tensor: torch.Tensor) -> np.ndarray:
    """
    frame_tensor: (C, H, W), normalized to [-1, 1]
    returns: (H, W, 3) in [0, 1]
    """
    img = frame_tensor.permute(1, 2, 0).detach().cpu().numpy()
    img = (img + 1.0) / 2.0
    return np.clip(img, 0.0, 1.0)


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam - cam.min()
    maxv = cam.max()
    if maxv > 0:
        cam = cam / maxv
    return cam


def overlay_heatmap(img: np.ndarray, cam: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    cam = normalize_cam(cam)
    heat = plt.get_cmap("jet")(cam)[..., :3]
    out = (1 - alpha) * img + alpha * heat
    return np.clip(out, 0.0, 1.0)


def cam_from_target(
    target: torch.Tensor,
    grads: torch.Tensor,
    hp: int,
    wp: int,
) -> List[np.ndarray]:
    """
    target/grads can be:
      (B, T, Hp, Wp, C) or
      (B, T, N, C) where N = Hp*Wp
    """
    cams: List[np.ndarray] = []
    _, T = target.shape[:2]

    for t in range(T):
        if target.dim() == 5:
            a = target[0, t] # (Hp, Wp, C)
            g = grads[0, t]  # (Hp, Wp, C)
        elif target.dim() == 4:
            a = target[0, t].reshape(hp, wp, -1)
            g = grads[0, t].reshape(hp, wp, -1)
        else:
            raise ValueError(f"Unsupported target dim: {target.dim()}")

        w = g.mean(dim=(0, 1))
        cam = F.relu((a * w).sum(dim=-1))

        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=FRAME_HW,
            mode="bilinear",
            align_corners=False
        ).squeeze().detach().cpu().numpy()

        cams.append(cam)

    return cams


def forward_variant_for_gradcam(
    model: torch.nn.Module,
    variant: str,
    video_b: torch.Tensor,
    roi_b: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Returns:
        logit: (B,)
        target tensor for CAM
        hp, wp
    """
    feats, flat = model._get_features(video_b) # feats:(B,T,Hp,Wp,C), flat:(B,T,N,C)
    hp, wp = feats.shape[2], feats.shape[3]

    if variant == "M1":
        target = feats
        target.retain_grad()

        cls_tokens = flat.mean(dim=2) # (B,T,C)
        logit = model.classifier(cls_tokens=cls_tokens)

    elif variant in ["M2", "M3"]:
        target = feats
        target.retain_grad()

        H_t, _ = model.tah(flat)
        logit = model.classifier(H=H_t)

    elif variant == "M4":
        with torch.autocast(device_type="cuda", enabled=False):
            F_roi, _ = model.oaa_r(
                feats.float(),
                roi_b.float() if roi_b is not None else None
            )
        F_roi = F_roi.to(feats.dtype)
        target = F_roi
        target.retain_grad()

        H_t, frame_scores = model.tah(F_roi)
        smoothed = model.smoother(frame_scores)
        logit = smoothed.mean(dim=1)

    elif variant == "M5":
        with torch.autocast(device_type="cuda", enabled=False):
            F_roi, _ = model.oaa_r(
                feats.float(),
                roi_b.float() if roi_b is not None else None
            )
        F_roi = F_roi.to(feats.dtype)
        target = F_roi
        target.retain_grad()

        H_t, frame_scores = model.tah(F_roi)
        smoothed = model.smoother(frame_scores)
        logit = smoothed.mean(dim=1)

    else:
        raise ValueError(f"Unsupported variant: {variant}")

    return logit, target, hp, wp


@torch.enable_grad()
def compute_gradcam(
    model: torch.nn.Module,
    variant: str,
    video_b: torch.Tensor,
    roi_b: Optional[torch.Tensor],
) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
    model.eval()
    model.zero_grad(set_to_none=True)

    logit, target, hp, wp = forward_variant_for_gradcam(model, variant, video_b, roi_b)
    prob = torch.sigmoid(logit).item()

    logit.backward(torch.ones_like(logit))

    grads = target.grad
    if grads is None:
        raise RuntimeError(f"No gradients captured for variant {variant}")

    cams = cam_from_target(target, grads, hp, wp)

    frames_rgb: List[np.ndarray] = []
    overlays: List[np.ndarray] = []

    T_video = video_b.shape[1]
    T_cam = len(cams)
    mapping = np.linspace(0, T_cam - 1, T_video)

    for t in range(T_video):
        frame = video_b[0, t]   # (C, H, W)
        img = denorm_frame(frame)

        cam_float = mapping[t]
        lo = min(int(np.floor(cam_float)), T_cam - 1)
        hi = min(int(np.ceil(cam_float)), T_cam - 1)

        if lo == hi:
            cam = cams[lo]
        else:
            a = cam_float - lo
            cam = (1 - a) * cams[lo] + a * cams[hi]

        over = overlay_heatmap(img, cam, alpha=ALPHA)

        frames_rgb.append(img)
        overlays.append(over)

    return frames_rgb, overlays, prob


def save_video(frames_uint8: List[np.ndarray], out_path: str, fps: int) -> None:
    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )
    for f in frames_uint8:
        writer.append_data(f)
    writer.close()
    print(f"  [saved] {out_path}")


def save_gif(frames_uint8: List[np.ndarray], out_path: str, fps: int) -> None:
    duration = 1.0 / fps
    imageio.mimsave(out_path, frames_uint8, duration=duration, loop=0)
    print(f"  [saved] {out_path}")


def make_side_by_side_frame(
    raw: np.ndarray,
    over: np.ndarray,
    variant: str,
    case_name: str,
    label: int,
    prob: float,
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    raw_u8 = (np.clip(raw, 0.0, 1.0) * 255).astype(np.uint8)
    over_u8 = (np.clip(over, 0.0, 1.0) * 255).astype(np.uint8)

    combined = np.concatenate([raw_u8, over_u8], axis=1)
    combined[:, raw_u8.shape[1], :] = 255

    img = Image.fromarray(combined)
    draw = ImageDraw.Draw(img)

    label_str = "Abnormal" if label == 1 else "Normal"
    header = (
        f"{variant} | {case_name} | GT: {label_str} | "
        f"Pred prob: {prob:.3f} | Frame {frame_idx+1}/{total_frames}"
    )

    draw.rectangle([0, 0, combined.shape[1], 20], fill=(0, 0, 0))
    draw.text((5, 3), header, fill=(255, 255, 255))

    draw.text((5, 24), "Original echo", fill=(255, 255, 255))
    draw.text((raw_u8.shape[1] + 10, 24), "Grad-CAM overlay", fill=(255, 255, 255))

    return np.array(img)


def save_case_outputs(
    frames_rgb: List[np.ndarray],
    overlays: List[np.ndarray],
    variant: str,
    case_name: str,
    label: int,
    prob: float,
    out_subdir: str,
) -> None:
    os.makedirs(out_subdir, exist_ok=True)

    combined_frames: List[np.ndarray] = []
    total = len(frames_rgb)

    for i, (raw, over) in enumerate(zip(frames_rgb, overlays)):
        combined = make_side_by_side_frame(
            raw, over, variant, case_name, label, prob, i, total
        )
        combined_frames.append(combined)

    mp4_path = os.path.join(out_subdir, f"{variant.lower()}_{case_name.lower()}_side_by_side.mp4")
    save_video(combined_frames, mp4_path, FPS)

    if SAVE_GIF:
        gif_path = os.path.join(out_subdir, f"{variant.lower()}_{case_name.lower()}_side_by_side.gif")
        save_gif(combined_frames, gif_path, GIF_FPS)


def main() -> None:
    ds = PediatricDataset(data_dir=DATA_DIR, split="TEST", augment=False)
    shared_cases = get_shared_cases()

    print("\nShared cases used for all variants:")
    for case_name, row in shared_cases.items():
        print(
            f"  {case_name}: idx={row['idx']} "
            f"label={row['label']} prob_ref={float(row['prob']):.4f}"
        )

    for variant in VARIANTS_TO_RUN:
        cfg = VARIANT_CONFIGS[variant]
        print(f"\n=== {variant} ===")

        model = load_model_for_variant(variant, str(cfg["exp_dir"]))
        rows_by_idx = {int(r["idx"]): r for r in load_rows(str(cfg["pred_csv"]))}

        for case_name, ref_row in shared_cases.items():
            idx = int(ref_row["idx"])
            if idx not in rows_by_idx:
                print(f"  [WARN] idx={idx} missing in {variant} prediction CSV, skipping.")
                continue

            row = rows_by_idx[idx]
            print(
                f"\n[{variant} | {case_name}] "
                f"idx={idx} label={row['label']} prob_csv={float(row['prob']):.4f}"
            )

            video, roi, _ = ds[idx]
            video_b = video.unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)
            roi_b = roi.unsqueeze(0).to(DEVICE) if roi is not None else None

            frames_rgb, overlays, prob = compute_gradcam(model, variant, video_b, roi_b)
            print(f"  clip_prob_model={prob:.4f} frames={len(frames_rgb)}")

            out_subdir = os.path.join(OUT_DIR, variant.lower(), f"{case_name.lower()}_{idx}")
            save_case_outputs(
                frames_rgb=frames_rgb,
                overlays=overlays,
                variant=variant,
                case_name=case_name,
                label=int(row["label"]),
                prob=prob,
                out_subdir=out_subdir,
            )

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()