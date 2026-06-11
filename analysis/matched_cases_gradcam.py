# Finds top N TP and FP candidates for M2_MTR for clinical plane review.
# Saves side-by-side MP4 videos (raw echo | GradCAM heatmap) for each candidate.

import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import glob
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from datasets.pediatric_dataset import EchoNetPediatricDataset as DS
from models.osa import build_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRED_CSV = "experiments/finetune_M3_v2/m3_test_preds.csv"
DATA_DIR = "data/echonet-pediatric"
EXP_DIR = "experiments/finetune_M3_v2"
OUT_DIR = "analysis/figures/gradcam_candidates"
THRESHOLD = 0.055
TOP_N = 10
VIDEO_FPS = 8

os.makedirs(f"{OUT_DIR}/tp", exist_ok=True)
os.makedirs(f"{OUT_DIR}/fp", exist_ok=True)

# Model loading 
def find_checkpoint(expdir):
    for c in [
        os.path.join(expdir, "best_m2mtr.pt"),
        os.path.join(expdir, "best.pt"),
    ]:
        if os.path.isfile(c):
            return c
    extra = sorted(glob.glob(os.path.join(expdir, "*.pt")))
    if extra:
        return extra[0]
    raise FileNotFoundError(f"No checkpoint in {expdir}")

def load_model():
    ckpt = find_checkpoint(EXP_DIR)
    model = build_model("M3", pretrained=True, freeze_core=True).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    if isinstance(state, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[M2_MTR] Loaded: {ckpt}")
    return model

# Predictions 
def load_preds(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "idx": int(row["idx"]),
                "label": int(float(row["label"])),
                "prob": float(row["prob"]),
                "path": row["path"],
            })
    return rows

# GradCAM
def compute_gradcam(model, video_b, roi_b):
    """Returns list of (H,W) CAM arrays (one per backbone token) and clip_prob."""
    feats = model.backbone(video_b) # (1,T,H',W',C')
    feats = feats.detach().requires_grad_(True)

    B, T_bb, Hp, Wp, C = feats.shape
    flat = feats.reshape(B, T_bb, Hp * Wp, C)
    H_t, _ = model.tah(flat)
    clip_logit = model.classifier(H=H_t)
    clip_prob = torch.sigmoid(clip_logit).item()

    model.zero_grad(set_to_none=True)
    clip_logit.backward(torch.ones_like(clip_logit))
    grads = feats.grad
    acts = feats

    cams = []
    for t in range(T_bb):
        g = grads[0, t]
        a = acts[0, t]
        w = g.mean(dim=(0, 1))
        cam = F.relu((a * w).sum(dim=-1))
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze().detach().cpu().numpy()
        cams.append(cam)
    return cams, clip_prob

def denorm(frame_tensor):
    img = frame_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img + 1.0) / 2.0
    return np.clip(img, 0, 1)

def overlay(img, cam, alpha=0.45):
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    heat = plt.get_cmap("jet")(cam)[..., :3]
    out  = (1 - alpha) * img + alpha * heat
    return np.clip(out, 0, 1)


# MP4 saving 

def save_side_by_side_mp4(raw_frames, cam_frames, out_path, fps,
                           case_type, idx, prob):
    try:
        import imageio
    except ImportError:
        print(" [WARN] imageio not installed. Run: pip install imageio imageio-ffmpeg")
        return

    from PIL import Image, ImageDraw

    combined_frames = []
    n = len(raw_frames)
    label_str = case_type

    for i, (raw, over) in enumerate(zip(raw_frames, cam_frames)):
        raw_u8  = (raw  * 255).astype(np.uint8)
        over_u8 = (over * 255).astype(np.uint8)

        # Stack side by side with 2px white divider
        divider = np.ones((224, 2, 3), dtype=np.uint8) * 200
        combined_np = np.concatenate([raw_u8, divider, over_u8], axis=1)

        img = Image.fromarray(combined_np)
        draw = ImageDraw.Draw(img)

        # Header bar
        draw.rectangle([0, 0, combined_np.shape[1], 20], fill=(20, 20, 20))
        draw.text(
            (4, 3),
            f"{label_str} | idx={idx} | prob={prob:.3f} | "
            f"frame {i+1}/{n}  ||  Raw          GradCAM",
            fill=(255, 255, 255),
        )
        combined_frames.append(np.array(img))

    # Write MP4
    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p"
    )
    for frame in combined_frames:
        writer.append_data(frame)
    writer.close()
    print(f"  [saved] {out_path}")


# Main 

def main():
    rows = load_preds(PRED_CSV)

    tp = sorted(
        [r for r in rows if r["label"] == 1 and r["prob"] >= THRESHOLD],
        key=lambda x: -x["prob"]
    )[:TOP_N]

    fp = sorted(
        [r for r in rows if r["label"] == 0 and r["prob"] >= THRESHOLD],
        key=lambda x: -x["prob"]
    )[:TOP_N]

    print(f"\nTop {TOP_N} True Positives:")
    for i, r in enumerate(tp):
        print(f"  rank {i+1:2d} | idx={r['idx']:<6} prob={r['prob']:.4f} | {r['path']}")

    print(f"\nTop {TOP_N} False Positives:")
    for i, r in enumerate(fp):
        print(f"  rank {i+1:2d} | idx={r['idx']:<6} prob={r['prob']:.4f} | {r['path']}")

    print(f"\nLoading model and dataset...")
    model = load_model()
    ds    = DS(data_dir=DATA_DIR, split="TEST", augment=False)

    for case_type, candidates, subdir in [
        ("TP", tp, f"{OUT_DIR}/tp"),
        ("FP", fp, f"{OUT_DIR}/fp"),
    ]:
        print(f"\nGenerating {case_type} MP4 videos...")
        for i, r in enumerate(candidates):
            idx = r["idx"]
            prob = r["prob"]
            video, roi, label = ds[idx]
            video_b = video.unsqueeze(0).to(DEVICE)
            roi_b = roi.unsqueeze(0).to(DEVICE)

            cams, clip_prob = compute_gradcam(model, video_b, roi_b)

            T_video = video.shape[0]
            T_bb = len(cams)
            mapping = np.linspace(0, T_bb - 1, T_video)

            raw_frames = []
            cam_frames = []

            for v_idx in range(T_video):
                raw = denorm(video[v_idx])

                bb_float = mapping[v_idx]
                bb_lo = int(math.floor(bb_float))
                bb_hi = int(math.ceil(bb_float))
                bb_lo = min(bb_lo, T_bb - 1)
                bb_hi = min(bb_hi, T_bb - 1)

                if bb_lo == bb_hi:
                    cam = cams[bb_lo]
                else:
                    alpha_interp = bb_float - bb_lo
                    cam = ((1 - alpha_interp) * cams[bb_lo] +
                            alpha_interp       * cams[bb_hi])

                raw_frames.append(raw)
                cam_frames.append(overlay(raw, cam))

            out_path = os.path.join(
                subdir,
                f"rank{i+1:02d}_idx{idx}_prob{prob:.3f}.mp4"
            )
            save_side_by_side_mp4(
                raw_frames, cam_frames, out_path,
                VIDEO_FPS, case_type, idx, clip_prob
            )

    print(f"\nAll MP4 videos saved to:")
    print(f"  {OUT_DIR}/tp/   — {TOP_N} true positive candidates")
    print(f"  {OUT_DIR}/fp/   — {TOP_N} false positive candidates")

if __name__ == "__main__":
    main()