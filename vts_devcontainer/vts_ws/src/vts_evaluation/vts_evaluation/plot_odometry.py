"""Plot ground truth against synchronized recorded wheel odometry.

Example:

    python3 -m vts_evaluation.plot_odometry \
        --gt-trajectory <sequence>/std_cam \
        --out output/revised/freiburg_a/images/path_compare.png
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np

from vts_core.matching import fit_se2
from vts_players.cold_odometry import load_cold_odometry
from vts_players.cid_sims_data import load_ground_truth, load_wheel_odometry

_FRAME = re.compile(
    r"t(?P<t>\d+\.\d+)_x(?P<x>-?\d+\.?\d*)_y(?P<y>-?\d+\.?\d*)"
)


def ground_truth(images_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Return COLD camera timestamps and ground-truth XY positions."""
    samples: list[tuple[float, float, float]] = []
    for name in os.listdir(images_dir):
        match = _FRAME.search(name)
        if match is not None:
            samples.append(
                (
                    float(match.group("t")),
                    float(match.group("x")),
                    float(match.group("y")),
                )
            )
    samples.sort()
    if not samples:
        raise RuntimeError(f"No parsable COLD poses in {images_dir}")
    values = np.asarray(samples, dtype=np.float64)
    return values[:, 0], values[:, 1:3]


def ground_truth_input(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load COLD ``std_cam`` or CID-SIMS ``groundtruth.txt``."""
    if os.path.isdir(path):
        return ground_truth(path)
    stream = load_ground_truth(path)
    return stream.timestamps, stream.poses[:, :2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-trajectory",
        required=True,
        help="COLD std_cam directory or CID-SIMS groundtruth.txt",
    )
    parser.add_argument(
        "--odom",
        help="odom.tdf/odom.txt; inferred from the ground-truth path",
    )
    parser.add_argument("--max-odometry-gap-s", type=float, default=1.0)
    parser.add_argument("--out", default="path_compare.png")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamps, gt = ground_truth_input(args.gt_trajectory)
    if args.odom:
        odom_path = args.odom
    elif os.path.isdir(args.gt_trajectory):
        odom_path = str(
            Path(args.gt_trajectory).parent / "odom_scans" / "odom.tdf"
        )
    else:
        odom_path = str(Path(args.gt_trajectory).parent / "odom.txt")
    is_cid_sims = Path(odom_path).name == "odom.txt"
    recorded = (
        load_wheel_odometry(odom_path)
        if is_cid_sims
        else load_cold_odometry(odom_path)
    )
    # Independent sensors rarely start and stop on exactly the same sample.
    # Evaluate only their temporal intersection instead of treating a few
    # ground-truth samples outside odometry coverage as malformed data.
    overlap = np.logical_and(
        timestamps >= recorded.timestamps[0],
        timestamps <= recorded.timestamps[-1],
    )
    timestamps = timestamps[overlap]
    gt = gt[overlap]
    if timestamps.size < 2:
        raise RuntimeError("Ground truth and odometry have no usable overlap")
    odom = np.asarray(
        [
            recorded.at(float(timestamp), args.max_odometry_gap_s)[0][:2]
            for timestamp in timestamps
        ],
        dtype=np.float64,
    )
    fit = fit_se2(odom, gt)
    if fit is None:
        raise RuntimeError("At least two poses are required for SE(2) alignment")
    rotation, translation = fit
    aligned_odom = odom @ rotation.T + translation

    errors = np.linalg.norm(aligned_odom - gt, axis=1)
    ate = float(np.sqrt(np.mean(errors**2)))
    median_error = float(np.median(errors))

    limits_low = np.minimum(gt.min(0), aligned_odom.min(0)) - 1.0
    limits_high = np.maximum(gt.max(0), aligned_odom.max(0)) + 1.0
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.plot(
        gt[:, 0], gt[:, 1], color="tab:green", linewidth=1.6,
        label="Ground truth",
    )
    axis.plot(
        aligned_odom[:, 0], aligned_odom[:, 1], color="tab:red", linewidth=1.4,
        label="Recorded wheel odometry",
    )
    axis.scatter(
        [gt[0, 0]], [gt[0, 1]], c="black", s=100, marker="*", label="Start",
        zorder=5,
    )
    axis.set(
        xlim=(limits_low[0], limits_high[0]),
        ylim=(limits_low[1], limits_high[1]),
        xlabel="x [m]",
        ylabel="y [m]",
        title=f"Recorded odometry: RMSE {ate:.2f} m, median {median_error:.2f} m",
    )
    axis.set_aspect("equal")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=14)
    figure.tight_layout()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"saved {output} | ATE_RMSE={ate:.2f} m median={median_error:.2f} m")


if __name__ == "__main__":
    main()
