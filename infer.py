from __future__ import annotations

import argparse

from src.engine.inference import run_inference
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for multimodal C3 YOLO detector.")
    parser.add_argument("--config", type=str, default="configs/c3_multimodal_yolo.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--save-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_inference(cfg, args.checkpoint, args.split, args.save_dir)


if __name__ == "__main__":
    main()
