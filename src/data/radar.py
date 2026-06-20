from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


RADAR_COLUMNS = [
    "timestamp",
    "range",
    "doppler",
    "azimuth",
    "elevation",
    "power",
    "x",
    "y",
    "z",
    "comp_height",
    "comp_velocity",
    "u",
    "v",
]


def load_radar_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in RADAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Radar csv missing columns: {missing}")
    return df


def _normalize(values: np.ndarray, clip_value: float) -> np.ndarray:
    if clip_value <= 0:
        return values.astype(np.float32)
    values = np.clip(values, -clip_value, clip_value) / clip_value
    return values.astype(np.float32)


def radar_points_to_grid(
    csv_path: str | Path,
    image_size: tuple[int, int],
    radar_cfg: Dict,
) -> np.ndarray:
    df = load_radar_csv(csv_path)
    height, width = image_size

    grid = np.zeros((4, height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    u = np.clip(df["u"].to_numpy(dtype=np.int32), 0, width - 1)
    v = np.clip(df["v"].to_numpy(dtype=np.int32), 0, height - 1)

    power = _normalize(df["power"].to_numpy(), float(radar_cfg.get("power_clip", 40.0)))
    doppler = _normalize(df["doppler"].to_numpy(), float(radar_cfg.get("doppler_clip", 15.0)))
    rng = np.clip(df["range"].to_numpy(dtype=np.float32), 0, float(radar_cfg.get("range_clip", 300.0)))
    rng = (rng / max(float(radar_cfg.get("range_clip", 300.0)), 1e-6)).astype(np.float32)
    height_feat = _normalize(df["comp_height"].to_numpy(), float(radar_cfg.get("range_clip", 300.0)))

    np.add.at(grid[0], (v, u), power)
    np.add.at(grid[1], (v, u), doppler)
    np.add.at(grid[2], (v, u), rng)
    np.add.at(grid[3], (v, u), height_feat)
    np.add.at(count, (v, u), 1.0)

    valid = count > 0
    for channel in range(grid.shape[0]):
        grid[channel, valid] /= count[valid]
    return grid
