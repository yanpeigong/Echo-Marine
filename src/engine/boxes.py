from __future__ import annotations

import torch


def box_xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    union = area1[:, None] + area2 - inter + 1e-7
    return inter / union


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, iou_type: str = "ciou") -> torch.Tensor:
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * (box1[..., 3] - box1[..., 1]).clamp(min=0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp(min=0) * (box2[..., 3] - box2[..., 1]).clamp(min=0)
    union = area1 + area2 - inter + 1e-7
    iou = inter / union

    if iou_type.lower() != "ciou":
        return iou

    c_x1 = torch.min(box1[..., 0], box2[..., 0])
    c_y1 = torch.min(box1[..., 1], box2[..., 1])
    c_x2 = torch.max(box1[..., 2], box2[..., 2])
    c_y2 = torch.max(box1[..., 3], box2[..., 3])
    c2 = (c_x2 - c_x1).pow(2) + (c_y2 - c_y1).pow(2) + 1e-7

    b1_x = (box1[..., 0] + box1[..., 2]) / 2
    b1_y = (box1[..., 1] + box1[..., 3]) / 2
    b2_x = (box2[..., 0] + box2[..., 2]) / 2
    b2_y = (box2[..., 1] + box2[..., 3]) / 2
    rho2 = (b2_x - b1_x).pow(2) + (b2_y - b1_y).pow(2)

    w1 = (box1[..., 2] - box1[..., 0]).clamp(min=1e-7)
    h1 = (box1[..., 3] - box1[..., 1]).clamp(min=1e-7)
    w2 = (box2[..., 2] - box2[..., 0]).clamp(min=1e-7)
    h2 = (box2[..., 3] - box2[..., 1]).clamp(min=1e-7)
    v = (4 / (torch.pi**2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    alpha = v / (1 - iou + v + 1e-7)
    ciou = iou - rho2 / c2 - alpha * v
    return ciou
