import cv2
import numpy as np
import torch
from typing import Optional, Tuple

def normalise_fps(frames: np.ndarray, src_fps: float, target_fps: float = 25.0) -> np.ndarray:
    """Resample (T, ...) array to target_fps via linear index interpolation."""
    T_src = len(frames)
    if T_src == 0:
        return frames
    duration = T_src / max(src_fps, 1.0)
    T_tgt = max(1, int(round(duration * target_fps)))
    idx = np.clip(np.linspace(0, T_src - 1, T_tgt).astype(int), 0, T_src - 1)
    return frames[idx]


def normalise_intensity(frames: np.ndarray) -> np.ndarray:
    """Normalise pixel values to [-1, 1] from uint8 [0, 255]."""
    return frames.astype(np.float32) / 127.5 - 1.0


def crop_and_resize(
    frames: np.ndarray, # (T, H, W) uint8 or float
    roi:    Optional[Tuple[int,int,int,int]], # (x, y, w, h) bounding box
    size:   int = 224,
) -> Tuple[np.ndarray, np.ndarray]:
    out = []
    for f in frames:
        if f.ndim == 3:
            f = f[..., 0]   # drop channel dim if present
        f = cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
        out.append(f)

    frames_out = np.stack(out).astype(np.float32) # (T, H, W) still raw
    # Build binary ROI mask at target resolution
    roi_mask = np.ones((size, size), dtype=np.float32)  # default: full frame
    if roi is not None:
        x, y, w, h = roi
        orig_H, orig_W = frames[0].shape[:2]
        x, y = max(0, x), max(0, y)
        w = min(w, orig_W - x)
        h = min(h, orig_H - y)
        x1 = int(round(x / orig_W * size))
        y1 = int(round(y / orig_H * size))
        x2 = int(round((x + w) / orig_W * size))
        y2 = int(round((y + h) / orig_H * size))
        roi_mask = np.zeros((size, size), dtype=np.float32)
        roi_mask[y1:y2, x1:x2] = 1.0
    return frames_out, roi_mask


def sample_clip(frames: np.ndarray, num_frames: int = 16) -> np.ndarray:
    """Uniformly sample exactly num_frames; tile if shorter than required."""
    T = len(frames)
    if T == 0:
        raise ValueError("Empty frame sequence passed to sample_clip.")
    if T < num_frames:
        reps = (num_frames // T) + 1
        frames = np.tile(frames, (reps,) + (1,) * (frames.ndim - 1))
    idx = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    return frames[idx]


def to_tensor(frames: np.ndarray, channels: int = 3) -> torch.Tensor:
    """
    (T, H, W) float32 → (T, C, H, W) float tensor.
    Grayscale is replicated to C channels if channels > 1.
    """
    t = torch.from_numpy(frames).float().unsqueeze(1) # (T, 1, H, W)
    if channels > 1:
        t = t.repeat(1, channels, 1, 1) # (T, 3, H, W)
    return t


def preprocess_video(
    frames: np.ndarray,
    src_fps: float,
    roi: Optional[Tuple[int,int,int,int]] = None,
    target_fps: float = 25.0,
    num_frames: int = 16,
    img_size:int = 224,
    channels: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full preprocessing pipeline for a single video.
    Returns:
        video: (T, C, H, W) float tensor in [-1, 1]
        mask: (1, H, W) float binary ROI mask
    """
    frames = normalise_fps(frames, src_fps, target_fps)
    frames = normalise_intensity(frames)
    frames, roi_mask = crop_and_resize(frames, roi, img_size)
    frames = sample_clip(frames, num_frames)
    video  = to_tensor(frames, channels)
    mask   = torch.from_numpy(roi_mask).unsqueeze(0)  # (1, H, W)
    return video, mask
