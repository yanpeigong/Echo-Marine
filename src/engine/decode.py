from __future__ import annotations

from typing import Dict, List

import torch


def make_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    yv, xv = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    return torch.stack((xv, yv), dim=-1).float()


def decode_level(
    level_pred: Dict[str, torch.Tensor],
    stride: int,
) -> Dict[str, torch.Tensor]:
    box = level_pred["box"]
    obj = level_pred["obj"]
    cls = level_pred["cls"]
    b, _, h, w = box.shape
    grid = make_grid(h, w, box.device).view(1, h, w, 2)

    box = box.permute(0, 2, 3, 1)
    obj = obj.permute(0, 2, 3, 1)
    cls = cls.permute(0, 2, 3, 1)

    xy = (box[..., 0:2].sigmoid() * 2.0 - 0.5 + grid) * stride
    wh = (box[..., 2:4].sigmoid() * 2.0).pow(2) * stride
    decoded = torch.cat([xy, wh], dim=-1).view(b, -1, 4)
    obj = obj.sigmoid().view(b, -1, 1)
    cls = cls.sigmoid().view(b, -1, cls.shape[-1])
    return {"boxes": decoded, "obj": obj, "cls": cls}


def decode_predictions(predictions: List[Dict[str, torch.Tensor]], strides: List[int]) -> Dict[str, torch.Tensor]:
    all_boxes = []
    all_obj = []
    all_cls = []
    for pred, stride in zip(predictions, strides):
        out = decode_level(pred, stride)
        all_boxes.append(out["boxes"])
        all_obj.append(out["obj"])
        all_cls.append(out["cls"])
    return {
        "boxes": torch.cat(all_boxes, dim=1),
        "obj": torch.cat(all_obj, dim=1),
        "cls": torch.cat(all_cls, dim=1),
    }
