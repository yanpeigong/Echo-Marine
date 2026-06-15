from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from src.models.backbone import TinyYOLOBackbone
from src.models.dehaze import AtmosphericPriorEstimator
from src.models.fusion import MultiScaleFusionNeck, QualityGatedFusion
from src.models.head import YOLOStyleHead


class MultimodalC3YOLO(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        width_mult = float(model_cfg.get("width_mult", 1.0))
        depth_mult = float(model_cfg.get("depth_mult", 1.0))
        self.num_classes = int(model_cfg["num_classes"])

        self.use_dehaze = bool(model_cfg.get("dehaze", {}).get("enabled", True))
        self.dehaze = AtmosphericPriorEstimator(hidden_dim=int(model_cfg["dehaze"].get("hidden_dim", 32)))

        self.rgb_backbone = TinyYOLOBackbone(3, width_mult, depth_mult)
        self.ir_backbone = TinyYOLOBackbone(1, width_mult, depth_mult)
        self.radar_backbone = TinyYOLOBackbone(
            int(cfg["dataset"]["radar"]["image_channels"]),
            width_mult,
            depth_mult,
        )

        feature_dims = self.rgb_backbone.out_channels
        hidden_dim = int(model_cfg["fusion"].get("hidden_dim", 256))
        heads = int(model_cfg["fusion"].get("attention_heads", 4))
        self.fusion_blocks = nn.ModuleList(
            [QualityGatedFusion(channels=c, hidden_dim=hidden_dim, num_heads=heads) for c in feature_dims]
        )

        neck_out = int(model_cfg["neck"]["out_channels"])
        self.neck = MultiScaleFusionNeck(feature_dims=feature_dims, out_channels=neck_out)
        self.head = YOLOStyleHead(
            num_classes=self.num_classes,
            feat_channels=int(model_cfg["head"]["feat_channels"]),
            stacked_convs=int(model_cfg["head"]["stacked_convs"]),
        )

        if neck_out != int(model_cfg["head"]["feat_channels"]):
            self.neck_adapt = nn.ModuleList(
                [nn.Conv2d(neck_out, int(model_cfg["head"]["feat_channels"]), 1) for _ in range(3)]
            )
        else:
            self.neck_adapt = None

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor, radar: torch.Tensor) -> Dict:
        aux = {}
        if self.use_dehaze:
            dehaze_outputs = self.dehaze(rgb)
            rgb_input = dehaze_outputs["enhanced"]
            aux["dehaze"] = dehaze_outputs
        else:
            rgb_input = rgb

        rgb_feats = self.rgb_backbone(rgb_input)
        ir_feats = self.ir_backbone(ir)
        radar_feats = self.radar_backbone(radar)

        fused_feats: List[torch.Tensor] = []
        quality_maps: List[torch.Tensor] = []
        for rgb_feat, ir_feat, radar_feat, fusion_block in zip(rgb_feats, ir_feats, radar_feats, self.fusion_blocks):
            fused, stats = fusion_block(rgb_feat, ir_feat, radar_feat)
            fused_feats.append(fused)
            quality_maps.append(stats["quality"])

        pyramid = self.neck(fused_feats)
        if self.neck_adapt is not None:
            pyramid = [layer(feat) for layer, feat in zip(self.neck_adapt, pyramid)]
        predictions = self.head(pyramid)
        aux["quality"] = quality_maps
        return {"predictions": predictions, "aux": aux}
