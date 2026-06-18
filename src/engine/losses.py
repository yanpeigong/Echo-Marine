from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.engine.boxes import bbox_iou, box_xywh_to_xyxy
from src.engine.decode import decode_predictions
from src.engine.matcher import build_targets
from src.models.dehaze import dehaze_regularization_loss


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = targets * prob + (1 - targets) * (1 - prob)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal = alpha_t * (1 - p_t).pow(self.gamma)
        return (bce * focal).mean()


class DetectionCriterion(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.loss_cfg = cfg["loss"]
        self.model_cfg = cfg["model"]
        self.num_classes = int(self.model_cfg["num_classes"])
        self.strides = list(self.model_cfg["strides"])
        self.cls_loss = FocalBCEWithLogitsLoss(
            alpha=float(self.loss_cfg.get("focal_alpha", 0.25)),
            gamma=float(self.loss_cfg.get("focal_gamma", 2.0)),
        )
        self.obj_loss = FocalBCEWithLogitsLoss(
            alpha=float(self.loss_cfg.get("focal_alpha", 0.25)),
            gamma=float(self.loss_cfg.get("focal_gamma", 2.0)),
        )

    def forward(self, outputs: Dict, targets: List[Dict], rgb_inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        predictions = outputs["predictions"]
        decoded = decode_predictions(predictions, self.strides)
        assigned = build_targets(decoded, targets, self.num_classes)

        pred_boxes = box_xywh_to_xyxy(decoded["boxes"])
        target_boxes = assigned["boxes"]
        fg_mask = assigned["fg_mask"]

        if fg_mask.any():
            ious = bbox_iou(pred_boxes[fg_mask], target_boxes[fg_mask], iou_type=self.loss_cfg.get("iou_type", "ciou"))
            box_loss = (1.0 - ious).mean()
        else:
            box_loss = pred_boxes.sum() * 0.0

        obj_logits = torch.cat(
            [pred["obj"].permute(0, 2, 3, 1).reshape(pred["obj"].shape[0], -1, 1) for pred in predictions],
            dim=1,
        )
        cls_logits = torch.cat(
            [pred["cls"].permute(0, 2, 3, 1).reshape(pred["cls"].shape[0], -1, self.num_classes) for pred in predictions],
            dim=1,
        )

        obj_loss = self.obj_loss(obj_logits, assigned["obj"])
        cls_loss = self.cls_loss(cls_logits, assigned["cls"])

        total = (
            float(self.loss_cfg.get("box_weight", 5.0)) * box_loss
            + float(self.loss_cfg.get("obj_weight", 1.0)) * obj_loss
            + float(self.loss_cfg.get("cls_weight", 1.0)) * cls_loss
        )

        dehaze_loss = box_loss.new_tensor(0.0)
        if "dehaze" in outputs["aux"] and self.model_cfg.get("dehaze", {}).get("enabled", True):
            dehaze_loss = dehaze_regularization_loss(outputs["aux"]["dehaze"], rgb_inputs)
            total = total + float(self.model_cfg["dehaze"].get("aux_loss_weight", 0.03)) * dehaze_loss

        return {
            "loss": total,
            "loss_box": box_loss.detach(),
            "loss_obj": obj_loss.detach(),
            "loss_cls": cls_loss.detach(),
            "loss_dehaze": dehaze_loss.detach(),
        }
