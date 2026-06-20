from __future__ import annotations

from typing import Dict

from torch.utils.data import DataLoader

from src.data.dataset import MultimodalDetectionDataset, detection_collate_fn


def build_dataloader(cfg: Dict, split: str, training: bool) -> DataLoader:
    dataset = MultimodalDetectionDataset(cfg, split=split, training=training)
    batch_size = int(cfg["train"]["batch_size"]) if training else max(1, int(cfg["train"]["batch_size"]) // 2)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=int(cfg["dataset"]["num_workers"]),
        pin_memory=True,
        drop_last=training,
        collate_fn=detection_collate_fn,
    )
