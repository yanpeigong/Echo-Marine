from __future__ import annotations

from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(kernel_size: int, padding: int | None = None, dilation: int = 1) -> int:
    if padding is not None:
        return padding
    return ((kernel_size - 1) * dilation) // 2


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding),
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = activation if activation is not None else nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, channels: int, hidden_ratio: float = 0.5) -> None:
        super().__init__()
        hidden = max(8, int(channels * hidden_ratio))
        self.cv1 = ConvBNAct(channels, hidden, 1)
        self.cv2 = ConvBNAct(hidden, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 2) -> None:
        super().__init__()
        self.cv1 = ConvBNAct(in_channels, out_channels, 1)
        self.blocks = nn.ModuleList([Bottleneck(out_channels) for _ in range(num_blocks)])
        self.cv2 = ConvBNAct(out_channels * (num_blocks + 1), out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        outputs = [x]
        current = x
        for block in self.blocks:
            current = block(current)
            outputs.append(current)
        return self.cv2(torch.cat(outputs, dim=1))


class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_size: int = 5) -> None:
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = ConvBNAct(in_channels, hidden, 1)
        self.pool = nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2)
        self.cv2 = ConvBNAct(hidden * 4, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class ResidualStack(nn.Module):
    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[Bottleneck(channels) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(F.gelu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


def make_divisible(v: int, divisor: int = 8) -> int:
    return int((v + divisor / 2) // divisor * divisor)


def upsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=ref.shape[-2:], mode="nearest")


def flatten_hw(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    return x.flatten(2).transpose(1, 2)


def restore_hw(x: torch.Tensor, hw: Iterable[int]) -> torch.Tensor:
    h, w = hw
    b, n, c = x.shape
    return x.transpose(1, 2).reshape(b, c, h, w)
