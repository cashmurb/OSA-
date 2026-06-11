# Generates side-by-side MP4 GradCAM videos for ResNet-50 and R3D-18 baselines
# Uses IDENTICAL clip indices as OSA GradCAM for direct cross-model comparison

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.video import r3d_18, R3D_18_Weights
from datasets.pediatric_dataset import EchoNetPediatricDataset as DS

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "data/echonet-pediatric"
OUT_DIR  = "analysis/figures/gradcam_baselines"
VIDEO_FPS = 8

# Update FP index once clinician confirms matched plane
FORCE_CASES = {
    "TP": 379,   
    "TN": 633,   
    "FP": 479,    
    "FN": 136,   
}

CONFIGS = {
    "ResNet50": {
        "ckpt": "experiments/baseline_resnet50/best_resnet50.pt",
        "preds": "experiments/baseline_resnet50/resnet50_test_preds.csv",
        "threshold": 0.7425,
        "outdir": os.path.join(OUT_DIR, "resnet50"),
    },
    "R3D18": {
        "ckpt": "experiments/baseline_r3d18/best_r3d18.pt",
        "preds": "experiments/baseline_r3d18/r3d18_test_preds.csv",
        "threshold": 0.2316,
        "outdir": os.path.join(OUT_DIR, "r3d18"),
    },
}


# Models 
class ResNet50Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(2048, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.features(x).squeeze(-1).squeeze(-1)
        feats = feats.view(B, T, 2048).mean(dim=1)
        return torch.sigmoid(self.classifier(feats)).squeeze(1)


class R3D18Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = r3d_18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        feats = self.features(x).squeeze(-1).squeeze(-1).squeeze(-1)
        return torch.sigmoid(self.classifier(feats)).squeeze(1)


def load_model(name, ckpt_path):
    if name == "ResNet50":
        model = ResNet50Baseline().to(DEVICE)
    else:
        model = R3D18Baseline().to(DEVICE)

    state = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[{name}] Loaded: {ckpt_path}")
    return model


# GradCAM 
def get_target_layer(model, name):
    """Last conv layer before global pool for each architecture."""
    if name == "ResNet50":
        # layer4 is index -2 in the Sequential (index -1 is AdaptiveAvgPool)
        return list(model.features.children())[-2][-1]
    else:
        # R3D-18: same structure
        return list(model.features.children())[-2][-1]


def compute_gradcam_resnet50(model, video_b):
    """Per-frame GradCAM for ResNet-50."""
    B, T, C, H, W = video_b.shape
    target_layer = get_target_layer(model, "ResNet50")
    cams, probs = [], []

    activations, gradients = {}, {}

    def fwd_hook(m, inp, out):
        activations["feat"] = out.detach()

    def bwd_hook(m, gin, gout):
        gradients["grad"] = gout[0].detach()

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    for t in range(T):
        frame = video_b[:, t].clone().requires_grad_(True)

        # Forward
        feat = model.features(frame).squeeze(-1).squeeze(-1)
        logit = model.classifier(feat)
        prob = torch.sigmoid(logit)
        probs.append(prob.item())

        # Backward
        model.zero_grad(set_to_none=True)
        logit.backward(torch.ones_like(logit))

        act = activations["feat"] # (1, C_feat, h, w)
        grd = gradients["grad"]  # (1, C_feat, h, w)
        w = grd.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((act * w).sum(dim=1))  # (1, h, w)
        cam = F.interpolate(
            cam.unsqueeze(1),
            size=(H, W), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
        cams.append(cam)

    fh.remove()
    bh.remove()
    return cams, float(np.mean(probs))


def compute_gradcam_r3d18(model, video_b):
    """Full-clip GradCAM for R3D-18 with temporal interpolation."""
    B, T, C, H, W = video_b.shape
    target_layer   = get_target_layer(model, "R3D18")

    activations, gradients = {}, {}

    def fwd_hook(m, inp, out):
        activations["feat"] = out

    def bwd_hook(m, gin, gout):
        gradients["grad"] = gout[0]

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    # R3D expects (B, C, T, H, W)
    x_3d = video_b.permute(0, 2, 1, 3, 4).contiguous()
    feat = model.features(x_3d)
    feat = feat.squeeze(-1).squeeze(-1).squeeze(-1)
    logit = model.classifier(feat)
    prob = torch.sigmoid(logit).item()

    model.zero_grad(set_to_none=True)
    logit.backward(torch.ones_like(logit))

    act = activations["feat"]   # (1, C_feat, T_feat, h, w)
    grd = gradients["grad"]     # (1, C_feat, T_feat, h, w)
    w = grd.mean(dim=(2, 3, 4), keepdim=True)
    cam = F.relu((act * w).sum(dim=1))  # (1, T_feat, h, w)

    T_feat = cam.shape[1]
    mapping = np.linspace(0, T_feat - 1, T)
    cams    = []

    for v_idx in range(T):
        bf   = mapping[v_idx]
        lo   = min(int(math.floor(bf)), T_feat - 1)
        hi   = min(int(math.ceil(bf)),  T_feat - 1)
        if lo == hi:
            cam_t = cam[0, lo]
        else:
            alpha = bf - lo
            cam_t = (1 - alpha) * cam[0, lo] + alpha * cam[0, hi]
        cam_up = F.interpolate(
            cam_t.unsqueeze(0).unsqueeze(0),
            size=(H, W), mode="bilinear", align_corners=False
        ).squeeze().detach().cpu().numpy()
        cams.append(cam_up)

    fh.remove()
    bh.remove()
    return cams, prob


# Visualisation 
def denorm(frame_tensor):
    img = frame_tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip((img + 1.0) / 2.0, 0, 1)


def overlay(img, cam, alpha=0.45):
    cam = cam - cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)


def save_mean_cam_png(raw_frames, cams, out_path, title):
    """Static mean GradCAM overlay — matches format of OSA cam_mean.png."""
    mean_cam = np.mean(cams, axis=0)
    mean_raw = np.mean(raw_frames, axis=0)
    if mean_cam.max() > 0:
        mean_cam = (mean_cam - mean_cam.min()) / mean_cam.max()
    heat = plt.get_cmap("jet")(mean_cam)[..., :3]
    over = np.clip(0.55 * mean_raw + 0.45 * heat, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(mean_raw, cmap="gray")
    axes[0].set_title("Raw (mean frame)", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(over)
    axes[1].set_title("GradCAM overlay", fontsize=10)
    axes[1].axis("off")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [saved] {out_path}")


def save_mp4(raw_frames, cam_frames, out_path, fps, header):
    """Side-by-side MP4."""
    try:
        import imageio
        from PIL import Image, ImageDraw
    except ImportError:
        print(" [WARN] imageio/Pillow not installed")
        return

    frames_out = []
    n = len(raw_frames)
    for i, (raw, over) in enumerate(zip(raw_frames, cam_frames)):
        raw_u8  = (raw  * 255).astype(np.uint8)
        over_u8 = (over * 255).astype(np.uint8)
        div     = np.ones((224, 2, 3), dtype=np.uint8) * 200
        combined = np.concatenate([raw_u8, div, over_u8], axis=1)
        img  = Image.fromarray(combined)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, combined.shape[1], 20], fill=(20, 20, 20))
        draw.text(
            (4, 3),
            f"{header} | frame {i+1}/{n}  ||  Raw          GradCAM",
            fill=(255, 255, 255),
        )
        frames_out.append(np.array(img))

    writer = imageio.get_writer(
        out_path, fps=fps, codec="libx264",
        quality=8, pixelformat="yuv420p"
    )
    for f in frames_out:
        writer.append_data(f)
    writer.close()
    print(f"    [saved] {out_path}")


# Main 
def main():
    ds = DS(data_dir=DATA_DIR, split="TEST", augment=False)

    for model_name, cfg in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")

        os.makedirs(cfg["outdir"], exist_ok=True)
        model = load_model(model_name, cfg["ckpt"])

        # Load preds for probability lookup
        preds = {}
        with open(cfg["preds"]) as f:
            for row in csv.DictReader(f):
                preds[int(row["idx"])] = float(row["prob"])

        for case_type, target_idx in FORCE_CASES.items():
            prob = preds.get(target_idx, -1.0)
            predicted_label = 1 if prob >= cfg["threshold"] else 0
            true_label = 1 if case_type in ("TP", "FN") else 0

            print(f"\n  {case_type} | idx={target_idx} | "
                  f"prob={prob:.4f} | pred={predicted_label} | "
                  f"true={true_label}")

            case_dir = os.path.join(
                cfg["outdir"],
                f"{case_type.lower()}_{target_idx}"
            )
            os.makedirs(case_dir, exist_ok=True)

            video, roi, label = ds[target_idx]
            video_b = video.unsqueeze(0).to(DEVICE)

            # GradCAM
            if model_name == "ResNet50":
                cams, clip_prob = compute_gradcam_resnet50(model, video_b)
            else:
                cams, clip_prob = compute_gradcam_r3d18(model, video_b)

            T = video.shape[0]
            raw_frames = [denorm(video[t]) for t in range(T)]
            cam_frames = [overlay(raw_frames[t], cams[t]) for t in range(T)]

            title  = (f"{model_name} | {case_type} | "
                      f"idx={target_idx} | prob={clip_prob:.3f}")
            header = (f"{model_name} | {case_type} | "
                      f"idx={target_idx} | prob={clip_prob:.3f}")

            # Static PNG — matches cam_mean.png convention
            save_mean_cam_png(
                raw_frames, cams,
                os.path.join(case_dir, "cam_mean.png"),
                title
            )

            # Side-by-side MP4
            save_mp4(
                raw_frames, cam_frames,
                os.path.join(case_dir, "cam_video.mp4"),
                VIDEO_FPS, header
            )

    print(f"\nAll outputs saved to: {OUT_DIR}/")
    print("\nStructure:")
    for name in CONFIGS:
        for ct, idx in FORCE_CASES.items():
            d = os.path.join(OUT_DIR, name.lower(),
                             f"{ct.lower()}_{idx}")
            print(f"  {d}/cam_mean.png")
            print(f"  {d}/cam_video.mp4")


if __name__ == "__main__":
    main()