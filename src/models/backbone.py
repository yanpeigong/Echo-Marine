from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.models.common import C2f, ConvBNAct, SPPF


class TinyYOLOBackbone(nn.Module):
    def __init__(self, in_channels: int, width_mult: float = 1.0, depth_mult: float = 1.0) -> None:
        super().__init__()
        c1 = int(64 * width_mult)
        c2 = int(128 * width_mult)
        c3 = int(256 * width_mult)
        c4 = int(512 * width_mult)
        d1 = max(1, int(2 * depth_mult))
        d2 = max(1, int(3 * depth_mult))
        d3 = max(1, int(3 * depth_mult))

        self.stem = ConvBNAct(in_channels, c1, 3, 2)
        self.stage1 = nn.Sequential(
            ConvBNAct(c1, c2, 3, 2),
            C2f(c2, c2, d1),
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(c2, c3, 3, 2),
            C2f(c3, c3, d2),
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(c3, c4, 3, 2),
            C2f(c4, c4, d3),
            SPPF(c4, c4),
        )

        self.out_channels = [c2, c3, c4]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)
        return [p3, p4, p5]
