from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.common import ConvBNAct, MLP, flatten_hw, restore_hw


class QualityEstimator(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionPromptEncoder(nn.Module):
    def __init__(self, channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor, radar: torch.Tensor) -> torch.Tensor:
        rgb_vec = self.pool(rgb).flatten(1)
        ir_vec = self.pool(ir).flatten(1)
        radar_vec = self.pool(radar).flatten(1)
        prompt = torch.cat([rgb_vec, ir_vec, radar_vec], dim=1)
        return self.mlp(prompt)


class FiLMModulator(nn.Module):
    def __init__(self, hidden_dim: int, channels: int) -> None:
        super().__init__()
        self.affine = nn.Linear(hidden_dim, channels * 2)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(prompt).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return x * (1.0 + 0.1 * gamma) + 0.1 * beta


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * 2)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q = self.norm1(query)
        kv = self.norm1(key_value)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        query = query + out
        query = query + self.mlp(self.norm2(query))
        return query


class QualityGatedFusion(nn.Module):
    def __init__(self, channels: int, hidden_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.rgb_proj = ConvBNAct(channels, hidden_dim, 1)
        self.ir_proj = ConvBNAct(channels, hidden_dim, 1)
        self.radar_proj = ConvBNAct(channels, hidden_dim, 1)
        self.condition_encoder = ConditionPromptEncoder(hidden_dim, hidden_dim)
        self.rgb_film = FiLMModulator(hidden_dim, hidden_dim)
        self.ir_film = FiLMModulator(hidden_dim, hidden_dim)
        self.radar_film = FiLMModulator(hidden_dim, hidden_dim)
        self.rgb_quality = QualityEstimator(hidden_dim)
        self.ir_quality = QualityEstimator(hidden_dim)
        self.radar_quality = QualityEstimator(hidden_dim)
        self.quality_bias = nn.Linear(hidden_dim, 3)
        self.rgb_ir_attn = CrossModalAttentionBlock(hidden_dim, num_heads=num_heads)
        self.rgb_radar_attn = CrossModalAttentionBlock(hidden_dim, num_heads=num_heads)
        self.radar_prior = nn.Sequential(
            ConvBNAct(hidden_dim, hidden_dim // 2, 3),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            ConvBNAct(hidden_dim * 3, hidden_dim, 1),
            ConvBNAct(hidden_dim, channels, 3),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        radar: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb = self.rgb_proj(rgb)
        ir = self.ir_proj(ir)
        radar = self.radar_proj(radar)
        prompt = self.condition_encoder(rgb, ir, radar)
        rgb = self.rgb_film(rgb, prompt)
        ir = self.ir_film(ir, prompt)
        radar = self.radar_film(radar, prompt)

        quality_bias = self.quality_bias(prompt)
        q_rgb = torch.sigmoid(self.rgb_quality(rgb) + quality_bias[:, 0:1, None, None])
        q_ir = torch.sigmoid(self.ir_quality(ir) + quality_bias[:, 1:2, None, None])
        q_radar = torch.sigmoid(self.radar_quality(radar) + quality_bias[:, 2:3, None, None])

        rgb_seq = flatten_hw(rgb)
        ir_seq = flatten_hw(ir)
        radar_seq = flatten_hw(radar)

        rgb_ir = self.rgb_ir_attn(rgb_seq, ir_seq)
        rgb_radar = self.rgb_radar_attn(rgb_seq, radar_seq)

        h, w = rgb.shape[-2:]
        rgb_ir = restore_hw(rgb_ir, (h, w))
        rgb_radar = restore_hw(rgb_radar, (h, w))
        radar_support = self.radar_prior(radar)
        rgb_ir = rgb_ir * (1.0 + 0.25 * q_ir)
        rgb_radar = rgb_radar * (1.0 + 0.5 * radar_support)
        rgb = rgb * (1.0 + 0.25 * q_rgb + 0.25 * radar_support)

        fused = torch.cat(
            [
                rgb * q_rgb,
                rgb_ir * q_ir,
                rgb_radar * q_radar,
            ],
            dim=1,
        )
        fused = self.out(fused)
        quality = torch.cat([q_rgb, q_ir, q_radar], dim=1)
        return fused, {"quality": quality, "condition_prompt": prompt, "radar_support": radar_support}


class MultiScaleFusionNeck(nn.Module):
    def __init__(self, feature_dims: List[int], out_channels: int) -> None:
        super().__init__()
        c3, c4, c5 = feature_dims
        self.lat5 = ConvBNAct(c5, out_channels, 1)
        self.lat4 = ConvBNAct(c4, out_channels, 1)
        self.lat3 = ConvBNAct(c3, out_channels, 1)
        self.out4 = ConvBNAct(out_channels * 2, out_channels, 3)
        self.out3 = ConvBNAct(out_channels * 2, out_channels, 3)
        self.down4 = ConvBNAct(out_channels, out_channels, 3, 2)
        self.pan4 = ConvBNAct(out_channels * 2, out_channels, 3)
        self.down5 = ConvBNAct(out_channels, out_channels, 3, 2)
        self.pan5 = ConvBNAct(out_channels * 2, out_channels, 3)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = features
        p5_lat = self.lat5(p5)
        p4_lat = self.lat4(p4)
        p3_lat = self.lat3(p3)

        p4_top = self.out4(torch.cat([p4_lat, F.interpolate(p5_lat, scale_factor=2, mode="nearest")], dim=1))
        p3_top = self.out3(torch.cat([p3_lat, F.interpolate(p4_top, scale_factor=2, mode="nearest")], dim=1))

        p4_pan = self.pan4(torch.cat([self.down4(p3_top), p4_top], dim=1))
        p5_pan = self.pan5(torch.cat([self.down5(p4_pan), p5_lat], dim=1))
        return [p3_top, p4_pan, p5_pan]
