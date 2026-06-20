from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.augment import eval_transform, train_transform
from src.data.radar import radar_points_to_grid


@dataclass
class SampleRecord:
    stem: str
    rgb_path: Path
    ir_path: Path
    radar_path: Path
    label_path: Path


def read_yolo_labels(path: Path, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image_size
    if not path.exists() or path.read_text(encoding="utf-8").strip() == "":
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    boxes = []
    labels = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        cls_id, cx, cy, bw, bh = map(float, line.split())
        cx *= width
        cy *= height
        bw *= width
        bh *= height
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        boxes.append([x1, y1, x2, y2])
        labels.append(int(cls_id))
    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


class MultimodalDetectionDataset(Dataset):
    def __init__(self, cfg: Dict, split: str, training: bool = True) -> None:
        self.cfg = cfg
        self.split = split
        self.training = training
        self.root = Path(cfg["dataset"]["processed_root"]) / split
        self.image_size = tuple(cfg["dataset"]["image_size"])
        self.aug_cfg = cfg.get("augmentation", {})
        self.radar_cfg = cfg["dataset"]["radar"]
        self.records = self._collect_records()

    def _collect_records(self) -> List[SampleRecord]:
        rgb_dir = self.root / "rgb"
        records = []
        for rgb_path in sorted(rgb_dir.glob("*.jpg")):
            stem = rgb_path.stem
            records.append(
                SampleRecord(
                    stem=stem,
                    rgb_path=rgb_path,
                    ir_path=self.root / "ir" / f"{stem}.png",
                    radar_path=self.root / "radar" / f"{stem}.csv",
                    label_path=self.root / "labels" / f"{stem}.txt",
                )
            )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: Path, mode: str) -> np.ndarray:
        img = Image.open(path).convert(mode)
        return np.array(img)

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]
        rgb = self._load_image(record.rgb_path, "RGB")
        ir = self._load_image(record.ir_path, "L")
        if ir.ndim == 2:
            ir = ir[..., None]

        orig_size = rgb.shape[:2]
        radar = radar_points_to_grid(record.radar_path, orig_size, self.radar_cfg)
        boxes, labels = read_yolo_labels(record.label_path, orig_size)

        sample = {
            "id": record.stem,
            "rgb": rgb,
            "ir": ir,
            "radar": radar,
            "boxes": boxes,
            "labels": labels,
            "orig_size": orig_size,
        }

        if self.training:
            sample = train_transform(sample, self.image_size, self.aug_cfg)
        else:
            sample = eval_transform(sample, self.image_size)

        rgb_tensor = torch.from_numpy(sample["rgb"]).permute(2, 0, 1).float() / 255.0
        ir_tensor = torch.from_numpy(sample["ir"]).permute(2, 0, 1).float() / 255.0
        radar_tensor = torch.from_numpy(sample["radar"]).float()
        boxes_tensor = torch.from_numpy(sample["boxes"]).float()
        labels_tensor = torch.from_numpy(sample["labels"]).long()

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.long),
            "orig_size": torch.tensor(sample["orig_size"], dtype=torch.long),
            "size": torch.tensor(sample["image_size"], dtype=torch.long),
            "sample_id": record.stem,
        }

        return {
            "rgb": rgb_tensor,
            "ir": ir_tensor,
            "radar": radar_tensor,
            "target": target,
        }


def detection_collate_fn(batch: List[Dict]) -> Dict:
    rgb = torch.stack([item["rgb"] for item in batch], dim=0)
    ir = torch.stack([item["ir"] for item in batch], dim=0)
    radar = torch.stack([item["radar"] for item in batch], dim=0)
    targets = [item["target"] for item in batch]
    return {"rgb": rgb, "ir": ir, "radar": radar, "targets": targets}
