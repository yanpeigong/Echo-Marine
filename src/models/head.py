from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.models.common import ConvBNAct


class DetectionTower(nn.Module):
    def __init__(self, channels: int, stacked_convs: int = 2) -> None:
        super().__init__()
        layers = []
        for _ in range(stacked_convs):
            layers.append(ConvBNAct(channels, channels, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class YOLOStyleHead(nn.Module):
    def __init__(self, num_classes: int, feat_channels: int, stacked_convs: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cls_towers = nn.ModuleList([DetectionTower(feat_channels, stacked_convs) for _ in range(3)])
        self.reg_towers = nn.ModuleList([DetectionTower(feat_channels, stacked_convs) for _ in range(3)])
        self.cls_preds = nn.ModuleList([nn.Conv2d(feat_channels, num_classes, 1) for _ in range(3)])
        self.obj_preds = nn.ModuleList([nn.Conv2d(feat_channels, 1, 1) for _ in range(3)])
        self.box_preds = nn.ModuleList([nn.Conv2d(feat_channels, 4, 1) for _ in range(3)])

    def forward(self, features: List[torch.Tensor]) -> List[dict]:
        outputs = []
        for idx, feat in enumerate(features):
            cls_feat = self.cls_towers[idx](feat)
            reg_feat = self.reg_towers[idx](feat)
            outputs.append(
                {
                    "cls": self.cls_preds[idx](cls_feat),
                    "obj": self.obj_preds[idx](reg_feat),
                    "box": self.box_preds[idx](reg_feat),
                }
            )
        return outputs
