"""Ground-truth trajectory loading shared by offline evaluation tools."""

from __future__ import annotations

import os
import re

import numpy as np

_COLD_FILENAME: re.Pattern[str] = re.compile(
    r"_x(?P<x>-?\d+\.?\d*)_y(?P<y>-?\d+\.?\d*)_a(?P<a>-?\d+\.?\d*)"
)


def _cold_trajectory(images_dir: str) -> np.ndarray:
    """Parse COLD ground-truth positions encoded in image filenames."""
    points: list[tuple[float, float]] = []
    for name in sorted(os.listdir(images_dir)):
        match: re.Match[str] | None = _COLD_FILENAME.search(name)
        if match is not None:
            points.append((float(match.group("x")), float(match.group("y"))))
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def load_ground_truth_xy(path: str) -> np.ndarray:
    """Load COLD image-name ground truth or a CID-SIMS ground-truth file."""
    if os.path.isdir(path):
        return _cold_trajectory(path)

    from vts_players.cid_sims_data import load_ground_truth

    return load_ground_truth(path).poses[:, :2].copy()
