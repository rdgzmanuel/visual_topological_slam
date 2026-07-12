"""Plot the ground-truth robot path vs the simulated drifting odometry.

Renders, side by side, the clean ground-truth trajectory (what the robot
actually did) and the simulated odometry the mapper consumes (ground truth +
the probabilistic motion-model drift), so the effect of the ``alpha`` noise is
visible directly. Both share the same start, so the red path's divergence from
the green is exactly the accumulated drift.

    python3 -m vts_evaluation.plot_odometry \
        --gt-trajectory <cold seq>/std_cam \
        --alpha 0.05 0.01 0.02 0.005 --odom-seed 17 \
        --out output/freiburg_a/images/path_compare.png

Uses only NumPy + matplotlib + vts_core.motion (no ROS, no torch). Pass the
same ``alpha`` / ``odom-seed`` your config uses to match the actual run.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np

from vts_core.motion import OdometryNoiseParams, OdometrySimulator

_POSE = re.compile(
    r"_x(?P<x>-?\d+\.?\d*)_y(?P<y>-?\d+\.?\d*)_a(?P<a>-?\d+\.?\d*)"
)


def gt_poses(images_dir: str) -> np.ndarray:
    """Parse (x, y, theta) ground-truth poses from COLD image filenames."""
    poses: list[tuple[float, float, float]] = []
    for name in sorted(os.listdir(images_dir)):
        match = _POSE.search(name)
        if match is not None:
            poses.append((
                float(match.group("x")),
                float(match.group("y")),
                float(match.group("a")),
            ))
    if not poses:
        raise RuntimeError(f"No parsable COLD poses in {images_dir}")
    return np.array(poses, dtype=np.float64)


def simulate_odometry(
    poses: np.ndarray, alpha: list[float], seed: int
) -> np.ndarray:
    """Run the odometry motion model over the GT poses; return (N, 2) xy."""
    sim = OdometrySimulator(OdometryNoiseParams(*alpha), seed=seed)
    out = np.array(
        [sim.step((float(p[0]), float(p[1]), float(p[2])))[0][:2] for p in poses]
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-trajectory", required=True, help="COLD std_cam dir")
    parser.add_argument(
        "--alpha", nargs=4, type=float, default=[0.05, 0.01, 0.02, 0.005],
        metavar=("A1", "A2", "A3", "A4"), help="odometry motion-model noise",
    )
    parser.add_argument("--odom-seed", type=int, default=17)
    parser.add_argument("--out", default="path_compare.png")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    poses = gt_poses(args.gt_trajectory)
    gt = poses[:, :2]
    odom = simulate_odometry(poses, args.alpha, args.odom_seed)

    error = np.linalg.norm(odom - gt, axis=1)  # same start -> raw comparison
    ate = float(np.sqrt(np.mean(error**2)))
    final_drift = float(error[-1])

    lim_lo = np.minimum(gt.min(0), odom.min(0)) - 1
    lim_hi = np.maximum(gt.max(0), odom.max(0)) + 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].plot(gt[:, 0], gt[:, 1], color="tab:green", lw=1.0)
    axes[0].set_title("ground truth (no odometry drift)")
    axes[1].plot(gt[:, 0], gt[:, 1], color="tab:green", lw=0.8, alpha=0.4,
                 label="ground truth")
    axes[1].plot(odom[:, 0], odom[:, 1], color="tab:red", lw=1.0,
                 label="simulated odometry")
    axes[1].set_title("with odometry drift")
    axes[1].legend(loc="best", fontsize=8)
    for ax in axes:
        ax.scatter([gt[0, 0]], [gt[0, 1]], c="black", s=70, marker="*", zorder=5)
        ax.set_aspect("equal")
        ax.set_xlim(lim_lo[0], lim_hi[0])
        ax.set_ylim(lim_lo[1], lim_hi[1])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"alpha={args.alpha}  seed={args.odom_seed}  |  "
        f"odometry ATE RMSE={ate:.2f} m, final drift={final_drift:.2f} m"
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(
        f"saved {args.out}  |  ATE_rmse={ate:.2f} m  final_drift={final_drift:.2f} m"
    )


if __name__ == "__main__":
    main()
