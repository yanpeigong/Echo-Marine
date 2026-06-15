from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import ConvBNAct


class AtmosphericPriorEstimator(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNAct(in_channels, hidden_dim, 3, 1),
            ConvBNAct(hidden_dim, hidden_dim, 3, 1),
            ConvBNAct(hidden_dim, hidden_dim, 3, 1),
        )
        self.transmission_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.airlight_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            ConvBNAct(in_channels + 1 + 3, hidden_dim, 3, 1),
            ConvBNAct(hidden_dim, hidden_dim, 3, 1),
            nn.Conv2d(hidden_dim, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.encoder(x)
        transmission = torch.sigmoid(self.transmission_head(feat))
        airlight = self.airlight_head(feat)
        airlight_map = airlight.expand(-1, -1, x.shape[-2], x.shape[-1])

        coarse = (x - airlight_map) / torch.clamp(transmission, min=0.1) + airlight_map
        coarse = coarse.clamp(0.0, 1.0)
        refined = self.refine(torch.cat([x, transmission, airlight_map], dim=1))
        enhanced = torch.clamp(0.7 * coarse + 0.3 * refined, 0.0, 1.0)

        return {
            "enhanced": enhanced,
            "transmission": transmission,
            "airlight": airlight,
            "coarse": coarse,
        }


def dehaze_regularization_loss(outputs: Dict[str, torch.Tensor], rgb: torch.Tensor) -> torch.Tensor:
    enhanced = outputs["enhanced"]
    transmission = outputs["transmission"]
    recon_loss = F.l1_loss(enhanced, rgb)

    dx = transmission[:, :, :, 1:] - transmission[:, :, :, :-1]
    dy = transmission[:, :, 1:, :] - transmission[:, :, :-1, :]
    smooth_loss = dx.abs().mean() + dy.abs().mean()

    contrast_orig = rgb.std(dim=(-2, -1)).mean()
    contrast_new = enhanced.std(dim=(-2, -1)).mean()
    contrast_gain = F.relu(contrast_orig - contrast_new)

    return recon_loss + 0.1 * smooth_loss + 0.05 * contrast_gain
