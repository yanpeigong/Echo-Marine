from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from src.data.build import build_dataloader
from src.engine.losses import DetectionCriterion
from src.engine.metrics import DetectionMeter
from src.engine.postprocess import postprocess_predictions
from src.models.multimodal_yolo import MultimodalC3YOLO
from src.utils.logger import setup_logger


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9998) -> None:
        self.ema = self._clone_model(model)
        self.decay = decay

    @staticmethod
    def _clone_model(model: torch.nn.Module) -> torch.nn.Module:
        ema = type(model)(model.cfg if hasattr(model, "cfg") else {})
        ema.load_state_dict(model.state_dict())
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    def to(self, device: torch.device) -> None:
        self.ema.to(device)

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
                else:
                    v.copy_(msd[k])


def build_optimizer(cfg: Dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = cfg["optimizer"]
    return AdamW(
        model.parameters(),
        lr=float(opt_cfg["lr"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )


def build_scheduler(cfg: Dict, optimizer: torch.optim.Optimizer) -> LambdaLR:
    warmup_epochs = int(cfg["scheduler"].get("warmup_epochs", 3))
    total_epochs = int(cfg["train"]["epochs"])
    min_lr_ratio = float(cfg["scheduler"].get("min_lr_ratio", 0.05))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return max(1e-3, (epoch + 1) / max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


class Trainer:
    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(cfg["project"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.output_dir)

        self.model = MultimodalC3YOLO(cfg)
        self.model.cfg = cfg
        self.model.to(self.device)
        self.criterion = DetectionCriterion(cfg)
        self.optimizer = build_optimizer(cfg, self.model)
        self.scheduler = build_scheduler(cfg, self.optimizer)
        self.scaler = GradScaler(enabled=bool(cfg["train"].get("mixed_precision", True)))
        self.ema = None
        if float(cfg["train"].get("ema_decay", 0.0)) > 0:
            self.ema = ModelEMA(self.model, decay=float(cfg["train"]["ema_decay"]))
            self.ema.to(self.device)

        self.train_loader = build_dataloader(cfg, "train", training=True)
        self.val_loader = build_dataloader(cfg, "val", training=False)
        self.best_f1 = -1.0

    def save_checkpoint(self, epoch: int, best: bool = False) -> None:
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_f1": self.best_f1,
            "config": self.cfg,
        }
        torch.save(state, self.output_dir / f"epoch_{epoch:03d}.pt")
        if best:
            torch.save(state, self.output_dir / "best.pt")

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meter = {"loss": 0.0, "loss_box": 0.0, "loss_obj": 0.0, "loss_cls": 0.0, "loss_dehaze": 0.0}
        progress = tqdm(self.train_loader, desc=f"train {epoch}", leave=False)
        for batch in progress:
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            ir = batch["ir"].to(self.device, non_blocking=True)
            radar = batch["radar"].to(self.device, non_blocking=True)
            targets = batch["targets"]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=bool(self.cfg["train"].get("mixed_precision", True))):
                outputs = self.model(rgb, ir, radar)
                losses = self.criterion(outputs, targets, rgb)
                loss = losses["loss"]

            self.scaler.scale(loss).backward()
            clip_value = float(self.cfg["train"].get("clip_grad_norm", 0.0))
            if clip_value > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update(self.model)

            for key in meter:
                meter[key] += float(losses[key].item())
            progress.set_postfix(loss=f"{meter['loss'] / max(1, progress.n):.4f}")

        num_batches = max(1, len(self.train_loader))
        for key in meter:
            meter[key] /= num_batches
        self.scheduler.step()
        return meter

    @torch.no_grad()
    def evaluate(self, use_ema: bool = True) -> Dict[str, float]:
        model = self.ema.ema if use_ema and self.ema is not None else self.model
        model.eval()
        meter = DetectionMeter(num_classes=int(self.cfg["model"]["num_classes"]))
        strides = list(self.cfg["model"]["strides"])
        infer_cfg = self.cfg["inference"]

        for batch in tqdm(self.val_loader, desc="eval", leave=False):
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            ir = batch["ir"].to(self.device, non_blocking=True)
            radar = batch["radar"].to(self.device, non_blocking=True)
            targets = batch["targets"]
            outputs = model(rgb, ir, radar)
            preds = postprocess_predictions(
                outputs["predictions"],
                strides=strides,
                conf_threshold=float(infer_cfg["conf_threshold"]),
                iou_threshold=float(infer_cfg["iou_threshold"]),
                max_detections=int(infer_cfg["max_detections"]),
            )
            cpu_targets = []
            for tgt in targets:
                cpu_targets.append({"boxes": tgt["boxes"].cpu(), "labels": tgt["labels"].cpu()})
            meter.update(preds, cpu_targets)
        return meter.compute()

    def fit(self) -> None:
        epochs = int(self.cfg["train"]["epochs"])
        for epoch in range(1, epochs + 1):
            train_stats = self.train_one_epoch(epoch)
            eval_stats = self.evaluate(use_ema=True)

            self.logger.info(
                "epoch=%03d loss=%.4f box=%.4f obj=%.4f cls=%.4f dehaze=%.4f precision=%.4f recall=%.4f f1=%.4f",
                epoch,
                train_stats["loss"],
                train_stats["loss_box"],
                train_stats["loss_obj"],
                train_stats["loss_cls"],
                train_stats["loss_dehaze"],
                eval_stats["precision"],
                eval_stats["recall"],
                eval_stats["f1"],
            )

            is_best = eval_stats["f1"] > self.best_f1
            if is_best:
                self.best_f1 = eval_stats["f1"]
            if is_best or epoch % int(self.cfg["train"].get("checkpoint_interval", 5)) == 0:
                self.save_checkpoint(epoch, best=is_best)
