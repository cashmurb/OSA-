import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from models.backbone_uniformer import UniFormerV2Backbone
from models.object_aware_head import OAAHead, OAAResidualHead
from models.temporal_head import TemporalAnomalyHead
from models.heads import TemporalSmoothingFilter, ClipClassifier
from configs.config import BACKBONE, OAA, TAH

class _OSABase(nn.Module):
    """Shared backbone. All variants extend this."""

    def __init__(self, pretrained: bool = True, freeze_core: bool = True):
        super().__init__()
        self.backbone = UniFormerV2Backbone(
            pretrained=pretrained,
            freeze_core=freeze_core,
        )
        self.C = self.backbone.feature_dim

    def _get_features(self, video: torch.Tensor):
        feats = self.backbone(video) # (B, T, H', W', C')
        B, T, Hp, Wp, C = feats.shape
        flat = feats.reshape(B, T, Hp * Wp, C)
        return feats, flat

    def forward(self, video, roi=None):
        raise NotImplementedError

class OSA_M1(_OSABase):
    """
    Baseline: backbone features globally pooled then classified.
    No TAH, no spatial prior, no smoothing.
    """

    def __init__(self, pretrained=True, freeze_core=True):
        super().__init__(pretrained, freeze_core)
        self.classifier = ClipClassifier(with_temporal=False, in_dim_cls=self.C)

    def forward(
        self,
        video: torch.Tensor,
        roi: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        feats, flat = self._get_features(video)
        cls_tokens = flat.mean(dim=2)
        clip_prob = self.classifier(cls_tokens=cls_tokens)
        return {
            "clip_prob": clip_prob,
            "frame_scores": None,
            "smoothed_scores": None,
            "oaa_weights": None,
        }

class OSA_M1_DEV(_OSABase):
    """
    Development variant: M1 + OAA head.
    Not included in reported experiments.
    """

    def __init__(self, pretrained=True, freeze_core=True):
        super().__init__(pretrained, freeze_core)
        self.oaa = OAAHead(feature_dim=self.C).float()  # FP32 to prevent AMP NaN
        self.classifier = ClipClassifier(with_temporal=False, in_dim_cls=self.C)

    def forward(self, video, roi=None):
        feats, _ = self._get_features(video)
        with torch.autocast(device_type="cuda", enabled=False):
            F_roi, oaa_w = self.oaa(feats.float(), roi.float() if roi is not None else None)
        F_roi = F_roi.to(feats.dtype)
        cls_tokens = F_roi.mean(dim=2)
        clip_prob = self.classifier(cls_tokens=cls_tokens)
        return {
            "clip_prob": clip_prob,
            "frame_scores": None,
            "smoothed_scores": None,
            "oaa_weights": oaa_w,
        }

class OSA_M2(_OSABase):
    """M1 + Temporal Aggregation and Anomaly Head, no spatial prior."""

    def __init__(self, pretrained=True, freeze_core=True):
        super().__init__(pretrained, freeze_core)
        self.tah = TemporalAnomalyHead(in_dim=self.C)
        self.classifier = ClipClassifier(with_temporal=True)

    def forward(self, video, roi=None):
        _, flat = self._get_features(video)
        H, frame_scores = self.tah(flat)
        clip_prob = self.classifier(H=H)
        return {
            "clip_prob": clip_prob,
            "frame_scores": frame_scores,
            "smoothed_scores": None,
            "oaa_weights": None,
        }

class OSA_M3(OSA_M2):
    """
    M2 with TAH initialised from MTR pre-training.
    Architecturally identical to M2 at fine-tuning time.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_core: bool = True,
        tah_pretrain_ckpt: str | None = None,
    ):
        super().__init__(pretrained=pretrained, freeze_core=freeze_core)
        if tah_pretrain_ckpt is not None:
            self._load_tah_pretrain(tah_pretrain_ckpt)

    def _load_tah_pretrain(self, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"TAH pretrain checkpoint not found: {ckpt_path}\n"
                f"Run training/tah_pretrain.py first."
            )
        state = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = self.tah.load_state_dict(state, strict=False)
        print(f"[M3] Loaded TAH pretrain weights from {ckpt_path}")
        if missing:
            print(f"[M3] Missing keys (expected: frame_scorer.*): {missing}")
        if unexpected:
            print(f"[M3] Unexpected keys: {unexpected}")
        for p in self.tah.recon_head.parameters():
            p.requires_grad_(False)

class OSA_M3_DEV(_OSABase):
    """
    Development variant: full hard OAA + TAH + smoothing.
    Superseded by M4/M5 in reported experiments.
    """

    def __init__(self, pretrained=True, freeze_core=True):
        super().__init__(pretrained, freeze_core)
        self.oaa = OAAHead(feature_dim=self.C).float()
        self.tah = TemporalAnomalyHead(in_dim=self.C)
        self.smoother = TemporalSmoothingFilter()
        self.classifier = ClipClassifier(with_temporal=True)

    def forward(self, video, roi=None):
        feats, _ = self._get_features(video)
        with torch.autocast(device_type="cuda", enabled=False):
            F_roi, oaa_w = self.oaa(feats.float(), roi.float() if roi is not None else None)
        F_roi = F_roi.to(feats.dtype)
        H, frame_scores = self.tah(F_roi)
        smoothed = self.smoother(frame_scores)
        clip_prob = smoothed.mean(dim=1)
        return {
            "clip_prob": clip_prob,
            "frame_scores": frame_scores,
            "smoothed_scores": smoothed,
            "oaa_weights": oaa_w,
        }

class OSA_M4(_OSABase):
    """
    Soft ROI residual bias + TAH + smoothing.
    Replaces hard OAA with multiplicative gating toward the cardiac ROI.
    """

    def __init__(self, pretrained=True, freeze_core=True):
        super().__init__(pretrained, freeze_core)
        self.oaa_r = OAAResidualHead(feature_dim=self.C).float()
        self.tah = TemporalAnomalyHead(in_dim=self.C)
        self.smoother = TemporalSmoothingFilter()

    def forward(self, video, roi=None):
        feats, _ = self._get_features(video)
        with torch.autocast(device_type="cuda", enabled=False):
            F_roi, _ = self.oaa_r(
                feats.float(),
                roi.float() if roi is not None else None,
            )
        F_roi = F_roi.to(feats.dtype)
        H, frame_scores = self.tah(F_roi)
        smoothed = self.smoother(frame_scores)
        clip_prob = smoothed.mean(dim=1)
        return {
            "clip_prob": clip_prob,
            "frame_scores": frame_scores,
            "smoothed_scores": smoothed,
            "oaa_weights": None,
        }

class OSA_M5(OSA_M4):
    """M4 with anatomically derived ROI masks from ventricular tracings."""
    pass


_REGISTRY = {
    "M1": OSA_M1,
    "M1_DEV": OSA_M1_DEV,
    "M2": OSA_M2,
    "M3": OSA_M3,
    "M3_DEV": OSA_M3_DEV,
    "M4": OSA_M4,
    "M5": OSA_M5,
}

def build_model(
    variant: str = "M1",
    pretrained: bool = True,
    freeze_core: bool = True,
    tah_pretrain_ckpt: str | None = None,
) -> nn.Module:
    key = variant.upper()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(_REGISTRY)}")

    if key == "M3":
        model = _REGISTRY[key](
            pretrained=pretrained,
            freeze_core=freeze_core,
            tah_pretrain_ckpt=tah_pretrain_ckpt,
        )
    else:
        model = _REGISTRY[key](pretrained=pretrained, freeze_core=freeze_core)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OSA-{key}] params: {total:,}  trainable: {trainable:,} ({100*trainable/total:.1f}%)")
    return model
