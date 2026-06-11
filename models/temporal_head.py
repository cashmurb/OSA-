import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from configs.config import TAH, BACKBONE

class MaskedTemporalReconHead(nn.Module):
    """
    Reconstruction head for MTR pretraining only. Projects encoder output H back to input feature space C'.
    Discarded after pretraining.
    """
    def __init__(
        self,
        proj_dim: int = TAH["proj_dim"],
        in_dim: int = BACKBONE["feature_dim"],
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, in_dim),
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.mlp(H)

def make_temporal_mask(B: int, T: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    """Returns (B, T) bool mask. True = masked. At least 1 frame always visible."""
    num_masked = max(1, min(T - 1, int(mask_ratio * T)))
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(T, device=device)[:num_masked]
        mask[b, idx] = True
    return mask

class TemporalAnomalyHead(nn.Module):
    """
    Temporal Aggregation and Anomaly Head (TAH).
    Takes (B, T, N, C') patch features and produces:
      H            : (B, T, D)  contextualised frame embeddings
      frame_scores : (B, T)     per-frame anomaly logits
    """

    def __init__(
        self,
        in_dim: int = BACKBONE["feature_dim"],
        proj_dim: int = TAH["proj_dim"],
        num_heads: int = TAH["num_heads"],
        num_layers: int = TAH["num_layers"],
        ffn_dim: int = TAH["ffn_dim"],
        dropout: float = TAH["dropout"],
        max_frames: int = 512,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        self.proj = nn.Linear(in_dim, proj_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, proj_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # learned [MASK] token used during MTR pretraining
        self.mask_token = nn.Parameter(torch.zeros(1, 1, proj_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(proj_dim),
        )

        self.frame_scorer = nn.Linear(proj_dim, 1)
        # bias init: sigmoid(-2) ≈ 0.12, keeps scores near zero at start
        nn.init.constant_(self.frame_scorer.bias, TAH["bias_init"])

        # recon_head is only used during MTR pretraining
        self.recon_head = MaskedTemporalReconHead(proj_dim=proj_dim, in_dim=in_dim)

    def _encode(self, F_roi: torch.Tensor, masked: bool = False, mask_ratio: float = 0.25):
        """Shared spatial pooling, projection, positional embedding, and encoding."""
        B, T = F_roi.shape[:2]

        V = F_roi.mean(dim=2) # spatial mean pool → (B, T, C')
        x = self.proj(V) # project → (B, T, D)

        pos = self.pos_embed[:, :T, :]
        if pos.shape[1] != T:
            pos = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=T, mode="linear", align_corners=False,
            ).permute(0, 2, 1)
        x = x + pos

        if masked:
            mask = make_temporal_mask(B, T, mask_ratio, device=F_roi.device)
            x_in = x.clone()
            x_in[mask] = self.mask_token.expand(B, T, -1)[mask]
            return V, self.encoder(x_in), mask

        return self.encoder(x)

    def forward(self, F_roi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        H = self._encode(F_roi)
        frame_scores = self.frame_scorer(H).squeeze(-1)
        return H, frame_scores

    def forward_mtr(
        self,
        F_roi: torch.Tensor,
        mask_ratio: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MTR forward. Called only by tah_pretrain.py."""
        V_orig, H, mask = self._encode(F_roi, masked=True, mask_ratio=mask_ratio)
        V_hat = self.recon_head(H)
        return V_orig, V_hat, mask
