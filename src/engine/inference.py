from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from src.data.build import build_dataloader
from src.engine.postprocess import postprocess_predictions
from src.models.multimodal_yolo import MultimodalC3YOLO


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)


@torch.no_grad()
def run_inference(cfg: Dict, checkpoint_path: str | Path, split: str, save_dir: str | Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalC3YOLO(cfg).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    loader = build_dataloader(cfg, split=split, training=False)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    class_names = cfg["dataset"]["class_names"]

    for batch in tqdm(loader, desc=f"infer {split}"):
        rgb = batch["rgb"].to(device)
        ir = batch["ir"].to(device)
        radar = batch["radar"].to(device)
        outputs = model(rgb, ir, radar)
        results = postprocess_predictions(
            outputs["predictions"],
            strides=list(cfg["model"]["strides"]),
            conf_threshold=float(cfg["inference"]["conf_threshold"]),
            iou_threshold=float(cfg["inference"]["iou_threshold"]),
            max_detections=int(cfg["inference"]["max_detections"]),
        )

        for rgb_tensor, pred, tgt in zip(batch["rgb"], results, batch["targets"]):
            image = (rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
            pil = Image.fromarray(image)
            draw = ImageDraw.Draw(pil)
            for box, score, label in zip(pred["boxes"].cpu(), pred["scores"].cpu(), pred["labels"].cpu()):
                x1, y1, x2, y2 = box.tolist()
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, max(0, y1 - 12)), f"{class_names[int(label)]}:{float(score):.2f}", fill="yellow")
            pil.save(save_dir / f"{tgt['sample_id']}.jpg")
