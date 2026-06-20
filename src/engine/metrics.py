from __future__ import annotations

from typing import Dict, List

import torch

from src.engine.boxes import box_iou


class DetectionMeter:
    def __init__(self, num_classes: int, iou_threshold: float = 0.5) -> None:
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self) -> None:
        self.tp = torch.zeros(self.num_classes, dtype=torch.float64)
        self.fp = torch.zeros(self.num_classes, dtype=torch.float64)
        self.fn = torch.zeros(self.num_classes, dtype=torch.float64)

    def update(self, predictions: List[Dict], targets: List[Dict]) -> None:
        for pred, tgt in zip(predictions, targets):
            gt_boxes = tgt["boxes"]
            gt_labels = tgt["labels"]
            pred_boxes = pred["boxes"].detach().cpu()
            pred_labels = pred["labels"].detach().cpu()

            matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
            for pb, pl in zip(pred_boxes, pred_labels):
                mask = gt_labels == pl
                candidate_idx = torch.where(mask)[0]
                if candidate_idx.numel() == 0:
                    self.fp[int(pl)] += 1
                    continue
                ious = box_iou(pb.unsqueeze(0), gt_boxes[candidate_idx])[0]
                best = torch.argmax(ious)
                if ious[best] >= self.iou_threshold and not matched[candidate_idx[best]]:
                    self.tp[int(pl)] += 1
                    matched[candidate_idx[best]] = True
                else:
                    self.fp[int(pl)] += 1
            for idx, gl in enumerate(gt_labels):
                if not matched[idx]:
                    self.fn[int(gl)] += 1

    def compute(self) -> Dict[str, float]:
        precision = self.tp / (self.tp + self.fp + 1e-9)
        recall = self.tp / (self.tp + self.fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        return {
            "precision": float(precision.mean().item()),
            "recall": float(recall.mean().item()),
            "f1": float(f1.mean().item()),
        }
