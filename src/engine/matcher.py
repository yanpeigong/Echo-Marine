from __future__ import annotations

from typing import Dict, List

import torch

from src.engine.boxes import box_iou, box_xywh_to_xyxy


def build_targets(
    decoded: Dict[str, torch.Tensor],
    targets: List[Dict],
    num_classes: int,
    topk: int = 10,
) -> Dict[str, torch.Tensor]:
    pred_boxes = box_xywh_to_xyxy(decoded["boxes"])
    pred_obj = decoded["obj"]
    pred_cls = decoded["cls"]

    batch_size, num_preds = pred_boxes.shape[:2]
    device = pred_boxes.device

    target_boxes = torch.zeros((batch_size, num_preds, 4), device=device)
    target_obj = torch.zeros((batch_size, num_preds, 1), device=device)
    target_cls = torch.zeros((batch_size, num_preds, num_classes), device=device)
    fg_mask = torch.zeros((batch_size, num_preds), dtype=torch.bool, device=device)

    for b in range(batch_size):
        gt_boxes = targets[b]["boxes"].to(device)
        gt_labels = targets[b]["labels"].to(device)
        if gt_boxes.numel() == 0:
            continue

        ious = box_iou(gt_boxes, pred_boxes[b])
        cls_scores = pred_cls[b][:, gt_labels].transpose(0, 1)
        cost = 1.0 - ious + (1.0 - cls_scores)

        for gt_idx in range(gt_boxes.shape[0]):
            k = min(topk, num_preds)
            top_idx = torch.topk(-cost[gt_idx], k=k, dim=0).indices
            best_idx = top_idx[ious[gt_idx, top_idx] > 0.1]
            if best_idx.numel() == 0:
                best_idx = top_idx[:1]
            target_boxes[b, best_idx] = gt_boxes[gt_idx].unsqueeze(0)
            target_obj[b, best_idx] = 1.0
            target_cls[b, best_idx, gt_labels[gt_idx]] = 1.0
            fg_mask[b, best_idx] = True

    return {
        "boxes": target_boxes,
        "obj": target_obj,
        "cls": target_cls,
        "fg_mask": fg_mask,
    }
