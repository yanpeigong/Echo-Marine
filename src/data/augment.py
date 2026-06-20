from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
from PIL import Image


def resize_image(image: np.ndarray, size: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
    pil = Image.fromarray(image)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    resized = pil.resize((size[1], size[0]), resample=resample)
    return np.array(resized)


def resize_boxes(
    boxes: np.ndarray,
    old_size: Tuple[int, int],
    new_size: Tuple[int, int],
) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    old_h, old_w = old_size
    new_h, new_w = new_size
    scale_x = new_w / old_w
    scale_y = new_h / old_h
    boxes = boxes.copy()
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    return boxes


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[:, ::-1])


def horizontal_flip_boxes(boxes: np.ndarray, width: int) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    boxes = boxes.copy()
    x1 = boxes[:, 0].copy()
    x2 = boxes[:, 2].copy()
    boxes[:, 0] = width - x2
    boxes[:, 2] = width - x1
    return boxes


def apply_hsv_jitter(image: np.ndarray, gains: Tuple[float, float, float]) -> np.ndarray:
    import cv2

    h_gain, s_gain, v_gain = gains
    r = np.random.uniform(-1, 1, 3) * np.array([h_gain, s_gain, v_gain]) + 1.0
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = np.mod(hsv[..., 0] * r[0], 180)
    hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def train_transform(
    sample: Dict,
    image_size: Tuple[int, int],
    aug_cfg: Dict,
) -> Dict:
    rgb = sample["rgb"]
    ir = sample["ir"]
    radar = sample["radar"]
    boxes = sample["boxes"]
    labels = sample["labels"]

    old_size = rgb.shape[:2]
    rgb = resize_image(rgb, image_size)
    ir = resize_image(ir, image_size)
    radar_resized = np.stack(
        [resize_image(ch, image_size) for ch in radar],
        axis=0,
    )
    boxes = resize_boxes(boxes, old_size, image_size)

    if random.random() < float(aug_cfg.get("horizontal_flip_prob", 0.0)):
        rgb = horizontal_flip(rgb)
        ir = horizontal_flip(ir)
        radar_resized = np.ascontiguousarray(radar_resized[:, :, ::-1])
        boxes = horizontal_flip_boxes(boxes, image_size[1])

    gains = aug_cfg.get("hsv_gain", [0.0, 0.0, 0.0])
    if max(gains) > 0:
        rgb = apply_hsv_jitter(rgb, tuple(gains))

    sample["rgb"] = rgb
    sample["ir"] = ir
    sample["radar"] = radar_resized
    sample["boxes"] = boxes
    sample["labels"] = labels
    sample["image_size"] = image_size
    return sample


def eval_transform(sample: Dict, image_size: Tuple[int, int]) -> Dict:
    rgb = sample["rgb"]
    ir = sample["ir"]
    radar = sample["radar"]
    boxes = sample["boxes"]
    old_size = rgb.shape[:2]

    sample["rgb"] = resize_image(rgb, image_size)
    sample["ir"] = resize_image(ir, image_size)
    sample["radar"] = np.stack([resize_image(ch, image_size) for ch in radar], axis=0)
    sample["boxes"] = resize_boxes(boxes, old_size, image_size)
    sample["image_size"] = image_size
    return sample
