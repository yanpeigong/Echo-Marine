from __future__ import annotations

from typing import Dict, List

import torch
import torchvision

from src.engine.boxes import box_xywh_to_xyxy
from src.engine.decode import decode_predictions


def postprocess_predictions(
    predictions: List[Dict[str, torch.Tensor]],
    strides: List[int],
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> List[Dict[str, torch.Tensor]]:
    decoded = decode_predictions(predictions, strides)
    boxes = box_xywh_to_xyxy(decoded["boxes"])
    obj = decoded["obj"]
    cls = decoded["cls"]

    results = []
    batch_size = boxes.shape[0]
    for b in range(batch_size):
        scores, labels = (obj[b] * cls[b]).max(dim=1)
        keep = scores > conf_threshold
        if keep.sum() == 0:
            results.append(
                {
                    "boxes": boxes[b].new_zeros((0, 4)),
                    "scores": boxes[b].new_zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.long, device=boxes.device),
                }
            )
            continue

        cur_boxes = boxes[b][keep]
        cur_scores = scores[keep]
        cur_labels = labels[keep]
        keep_idx = torchvision.ops.batched_nms(cur_boxes, cur_scores, cur_labels, iou_threshold)
        keep_idx = keep_idx[:max_detections]
        results.append(
            {
                "boxes": cur_boxes[keep_idx],
                "scores": cur_scores[keep_idx],
                "labels": cur_labels[keep_idx],
            }
        )
    return results
