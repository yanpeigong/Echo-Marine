from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.engine.trainer import Trainer
from src.utils.config import load_config
from src.utils.misc import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal C3 YOLO detector.")
    parser.add_argument("--config", type=str, default="configs/c3_multimodal_yolo.yaml")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    trainer = Trainer(cfg)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=trainer.device)
        trainer.model.load_state_dict(ckpt["model"], strict=True)

    if args.evaluate_only:
        stats = trainer.evaluate(use_ema=False)
        print(stats)
        return

    trainer.fit()


if __name__ == "__main__":
    main()
