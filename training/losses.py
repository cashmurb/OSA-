import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from configs.config import LOSS


class OSALoss(nn.Module):
    def __init__(
        self,
        lambda_reg: float = LOSS["lambda_reg"],
        lambda_smooth: float = LOSS["lambda_smooth"],
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_smooth = lambda_smooth
        pw = torch.tensor([pos_weight]).cuda() if pos_weight is not None else None
        self.bce_logits = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(
        self,
        clip_prob: torch.Tensor,
        labels: torch.Tensor,
        frame_scores: Optional[torch.Tensor],
        attn_maps: Optional[torch.Tensor],
        roi: Optional[torch.Tensor],
    ) -> dict:
        l_cls = self.bce_logits(clip_prob, labels)

        l_reg = torch.tensor(0.0, device=clip_prob.device)
        if attn_maps is not None and roi is not None:
            B, T, H, N, _ = attn_maps.shape
            side = int(N ** 0.5)
            M = F.interpolate(roi.float(), size=(side, side),
                              mode="bilinear", align_corners=False)
            M = M.view(B, N)
            outside = 1.0 - M
            attn_mean = attn_maps.mean(dim=(2, 3))
            outside_t = outside.unsqueeze(1).expand(B, T, N)
            l_reg = (attn_mean * outside_t).mean()

        l_smooth = torch.tensor(0.0, device=clip_prob.device)
        if frame_scores is not None and frame_scores.shape[1] > 1 and self.lambda_smooth > 0:
            diff = frame_scores[:, 1:] - frame_scores[:, :-1]
            l_smooth = (diff ** 2).mean()

        l_total = l_cls + self.lambda_reg * l_reg + self.lambda_smooth * l_smooth
        return {"total": l_total, "cls": l_cls, "reg": l_reg, "smooth": l_smooth}
