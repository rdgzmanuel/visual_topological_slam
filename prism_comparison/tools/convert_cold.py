"""Convert COLD sequences into self-contained input bundles for PRISM-TopoMap.

For each sequence this script produces a single ``<sequence>.npz`` bundle with:

- ``scan_t``      (N,)      SICK scan timestamps [s]
- ``scan_ranges`` (N, 181)  raw laser ranges [m] (181 beams, -90..+90 deg, 1 deg step)
- ``scan_gt``     (N, 3)    ground-truth pose (x, y, theta) interpolated to scan times
- ``scan_odom``   (N, 3)    synthesized odometry pose interpolated to scan times
- ``scan_label``  (N,)      room label of the nearest image frame (from places.lst)
- ``img_t``       (M,)      image timestamps [s]
- ``img_gt``      (M, 3)    ground-truth pose parsed from image filenames
- ``img_label``   (M,)      room labels per image
- ``meta``        json string with provenance (alphas, seed, angle convention)

Odometry is synthesized exactly as in the VTS experiments: the probabilistic
odometry motion model (vts_core.motion.OdometrySimulator) is stepped once per
IMAGE frame with the same noise coefficients and seed as the VTS configs, so
both systems receive the identical odometry realization. The resulting
piecewise trajectory is then linearly interpolated to the scan timestamps.

Run on the host (numpy only):
    python tools/convert_cold.py [--dataset-root PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MOTION_PY = (
    REPO_ROOT
    / "vts_devcontainer/vts_ws/src/vts_core/vts_core/motion.py"
)

# Same values as vts_bringup/config/cold_*.yaml (alpha1..4, odom_seed).
ALPHA = [0.025, 0.005, 0.01, 0.0025]
ODOM_SEED = 17

SEQUENCES = {
    "freiburg_a": "cold-freiburg_part_a_seq2_night1",
    "freiburg_ext": "cold-freiburg_part_b_seq3_sunny1",
    "saarbruecken_a": "cold-saarbruecken_part_a_seq2_night2",
    "saarbruecken_ext": "cold-saarbruecken_part_b_seq4_sunny1",
}

IMG_RE = re.compile(
    r"t(?P<t>\d+\.\d+)_x(?P<x>-?\d+\.\d+)_y(?P<y>-?\d+\.\d+)_a(?P<a>-?\d+\.\d+)\.jpeg"
)


def load_motion_module():
    spec = importlib.util.spec_from_file_location("vts_motion", MOTION_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vts_motion"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_places(seq_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Parse places.lst -> (img_t, img_gt, img_label). Sorted by time."""
    rows = []
    for line in (seq_dir / "localization" / "places.lst").read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        m = IMG_RE.match(parts[0])
        if m is None:
            continue
        rows.append(
            (float(m["t"]), float(m["x"]), float(m["y"]), float(m["a"]), parts[1])
        )
    rows.sort(key=lambda r: r[0])
    t = np.array([r[0] for r in rows])
    gt = np.array([[r[1], r[2], r[3]] for r in rows])
    labels = [r[4] for r in rows]
    return t, gt, labels


def parse_scans(seq_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse scans.tdf -> (scan_t, ranges[N, 181])."""
    ts, ranges = [], []
    for line in (seq_dir / "odom_scans" / "scans.tdf").read_text().splitlines():
        f = line.split()
        if len(f) < 17:
            continue
        n = int(f[2])
        sec, usec = int(f[3]), int(f[4])
        vals = [float(v) for v in f[len(f) - n :]]
        if len(vals) != n:
            continue
        ts.append(sec + usec * 1e-6)
        ranges.append(vals)
    t = np.array(ts)
    r = np.array(ranges)
    order = np.argsort(t)
    return t[order], r[order]


def interp_pose(query_t: np.ndarray, t: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Linear interpolation of (x, y, theta) with angle unwrapping."""
    x = np.interp(query_t, t, poses[:, 0])
    y = np.interp(query_t, t, poses[:, 1])
    theta_unwrapped = np.unwrap(poses[:, 2])
    theta = np.interp(query_t, t, theta_unwrapped)
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return np.stack([x, y, theta], axis=1)


def convert(env: str, seq_name: str, dataset_root: Path, out_dir: Path, motion) -> dict:
    seq_dir = dataset_root / seq_name
    img_t, img_gt, img_label = parse_places(seq_dir)
    scan_t, scan_ranges = parse_scans(seq_dir)

    # Keep only scans inside the ground-truth (image) time span.
    keep = (scan_t >= img_t[0]) & (scan_t <= img_t[-1])
    scan_t, scan_ranges = scan_t[keep], scan_ranges[keep]

    # Odometry synthesized per image frame — identical to the VTS runs.
    sim = motion.OdometrySimulator(
        params=motion.OdometryNoiseParams(*ALPHA), seed=ODOM_SEED
    )
    img_odom = np.array([sim.step(tuple(p))[0] for p in img_gt])

    scan_gt = interp_pose(scan_t, img_t, img_gt)
    scan_odom = interp_pose(scan_t, img_t, img_odom)
    nearest_img = np.searchsorted(img_t, scan_t)
    nearest_img = np.clip(nearest_img, 0, len(img_t) - 1)
    prev_img = np.clip(nearest_img - 1, 0, len(img_t) - 1)
    use_prev = np.abs(img_t[prev_img] - scan_t) < np.abs(img_t[nearest_img] - scan_t)
    nearest_img[use_prev] = prev_img[use_prev]
    scan_label = [img_label[i] for i in nearest_img]

    meta = {
        "environment": env,
        "sequence": seq_name,
        "alpha": ALPHA,
        "odom_seed": ODOM_SEED,
        "n_scans": int(len(scan_t)),
        "n_images": int(len(img_t)),
        "beam_count": int(scan_ranges.shape[1]),
        "angle_min_rad": -np.pi / 2,
        "angle_max_rad": np.pi / 2,
        "max_range_m": 81.9,
        "duration_s": float(scan_t[-1] - scan_t[0]),
        "gt_path_length_m": float(
            np.sum(np.hypot(np.diff(img_gt[:, 0]), np.diff(img_gt[:, 1])))
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{env}.npz",
        scan_t=scan_t,
        scan_ranges=scan_ranges,
        scan_gt=scan_gt,
        scan_odom=scan_odom,
        scan_label=np.array(scan_label),
        img_t=img_t,
        img_gt=img_gt,
        img_label=np.array(img_label),
        meta=json.dumps(meta),
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=REPO_ROOT / "encoder" / "seq_data"
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parents[1] / "data"
    )
    args = parser.parse_args()

    motion = load_motion_module()
    for env, seq_name in SEQUENCES.items():
        meta = convert(env, seq_name, args.dataset_root, args.out, motion)
        print(
            f"{env}: {meta['n_scans']} scans, {meta['n_images']} images, "
            f"{meta['gt_path_length_m']:.1f} m over {meta['duration_s']:.0f} s"
        )


if __name__ == "__main__":
    main()
