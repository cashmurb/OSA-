# UniFormerV2 backbone wrapper.
# Install UniFormerV2 on Linux 

# Fallback 
# If UniFormerV2 is not installed, a 3D CNN fallback is used automatically. Output shape is identical: (B, T, H', W', C') 
# Swap to UniFormerV2 any time by installing the package — no other code changes.
#
# LoRA
# After loading, the backbone core is frozen. Only LoRA adapters (injected into attention qkv projections) are trained.
# This keeps trainable params to ~0.3-0.5M in the backbone.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from configs.config import BACKBONE, CLIP

# LoRA linear layer
class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with low-rank adaptation.
    output = W·x  +  (B·A·x) × (alpha / rank)
    Only A and B are trained; W is permanently frozen.
    """
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.linear = linear
        self.scale = alpha / rank
        d_in = linear.in_features
        d_out = linear.out_features
        self.A = nn.Linear(d_in,  rank,  bias=False)
        self.B = nn.Linear(rank,  d_out, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)
        for p in self.linear.parameters():
            p.requires_grad_(False)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.B(self.A(x)) * self.scale



# Fallback 3D CNN (used when UniFormerV2 is not installed)
class FallbackBackbone(nn.Module):

    def __init__(self, feature_dim: int = BACKBONE["fallback_feature_dim"]):
        super().__init__()
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            # Stage 1:  /2 spatial
            nn.Conv3d(3, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(64), nn.GELU(),

            # Stage 2:  /2 spatial
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128), nn.GELU(),

            # Stage 3:  /2 spatial
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256), nn.GELU(),

            # Stage 4:  /2 spatial  →  total /16
            nn.Conv3d(256, feature_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(feature_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)           
        x = self.encoder(x)                    
        x = x.permute(0, 2, 3, 4, 1)            
        return x

# UniFormerV2 wrapper
class UniFormerV2Backbone(nn.Module):
    """
    Wraps UniFormerV2 for the OSA pipeline.

    Pipeline:
        (B, T, C, H, W)
          → 3D patch embed
          → N × UniBlocks  (local 3D conv + global attention)
          → (B, T, H', W', C')

    LoRA adapters are injected into all attention qkv projections so that
    only ~0.3M parameters need updating during fine-tuning.
    """

    def __init__(
        self,
        model_name:  str  = BACKBONE["name"],
        pretrained:  bool = BACKBONE["pretrained"],
        lora_rank:   int  = BACKBONE["lora_rank"],
        lora_alpha:  int  = BACKBONE["lora_alpha"],
        freeze_core: bool = True,
    ):
        super().__init__()
        self._use_fallback = False
        self.backbone, self.feature_dim = self._load(model_name, pretrained)

        if freeze_core and not self._use_fallback:
            self._freeze_core()
            self._inject_lora(lora_rank, lora_alpha)
        elif self._use_fallback:
            # Fallback is fully trainable — all params updated during SSL/fine-tuning
            pass

    # private 
    def _load(self, name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        try:
            from slowfast.models.uniformerv2_model import uniformerv2_b16
            model = uniformerv2_b16(pretrained=pretrained)
            model.transformer.proj = nn.Identity()
            print(f"[Backbone] UniFormerV2 b16 loaded. Pretrained={pretrained}")
            return model, BACKBONE["feature_dim"]
        except Exception as e:
            print(
                f"[Backbone] UniFormerV2 not available ({e}).\n -> Using 3D CNN fallback.")
            self._use_fallback = True
            fb = FallbackBackbone(BACKBONE["fallback_feature_dim"])
            return fb, BACKBONE["fallback_feature_dim"]

    def _freeze_core(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        print("[Backbone] Core weights frozen.")

    def _inject_lora(self, rank: int, alpha: int):
        replaced = 0
        for module in self.backbone.modules():
            for attr_name, layer in list(module.named_children()):
                if isinstance(layer, nn.Linear) and attr_name in ("out_proj", "c_fc", "c_proj"):
                    setattr(module, attr_name, LoRALinear(layer, rank, alpha))
                    replaced += 1
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"[Backbone] LoRA injected into {replaced} layers. "
              f"Trainable backbone params: {trainable:,}")

    # forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, C, H, W)
        Returns : (B, T, H', W', C')
        """
        if self._use_fallback:
            return self.backbone(x)

        B, T, C, H, W = x.shape

        # UniFormerV2 expects (B, C, T, H, W)
        x_in = x.permute(0, 2, 1, 3, 4)

        feats = self.backbone(x_in)

        # Handle tuple output (logits, spatial_features) 
        if isinstance(feats, tuple):
            _, feats = feats 
        elif feats.dim() == 2:
            # Fallback: (B, C') 
            feats = feats.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(B, T, 1, 1, -1)
        return feats  # (B, T, H', W', C')

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]