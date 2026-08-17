"""Pure-Python CID-SIMS trajectory readers and timestamp synchronization.

Official CID-SIMS sequence layout::

    color/<timestamp>.png
    groundtruth.txt  # timestamp px py pz qx qy qz qw
    odom.txt   # timestamp px py pz qx qy qz qw vx vy vz wx wy wz

Only planar pose components are exposed to the topological mapper. Ground
truth remains a separate evaluation stream and is never used for mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vts_core.motion import (
    OdometryUncertaintyParams,
    OdometryUncertaintyTracker,
    Pose2D,
    normalize_angle,
)


def quaternion_yaw(x: float, y: float, z: float, w: float) -> float:
    """Return the standard Z-axis yaw of a unit quaternion."""
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(sin_yaw, cos_yaw))


@dataclass(frozen=True)
class PlanarPoseStream:
    """Timestamped planar poses with optional cumulative uncertainty."""

    timestamps: np.ndarray
    poses: np.ndarray
    covariances: np.ndarray | None = None

    def at(
        self, timestamp: float, max_gap_s: float
    ) -> tuple[Pose2D, np.ndarray | None]:
        """Interpolate pose, angle and covariance at ``timestamp``."""
        if timestamp < self.timestamps[0] or timestamp > self.timestamps[-1]:
            raise ValueError(f"timestamp {timestamp:.9f} is outside pose coverage")
        upper = int(np.searchsorted(self.timestamps, timestamp, side="left"))
        if upper < len(self.timestamps) and np.isclose(
            self.timestamps[upper], timestamp, atol=1e-9, rtol=0.0
        ):
            covariance = (
                None
                if self.covariances is None
                else self.covariances[upper].copy()
            )
            return tuple(float(v) for v in self.poses[upper]), covariance
        if upper == 0 or upper == len(self.timestamps):
            raise ValueError(f"cannot bracket timestamp {timestamp:.9f}")

        lower = upper - 1
        gap = float(self.timestamps[upper] - self.timestamps[lower])
        if gap <= 0.0 or gap > max_gap_s:
            raise ValueError(
                f"pose gap {gap:.3f}s around {timestamp:.9f} exceeds "
                f"the {max_gap_s:.3f}s limit"
            )
        fraction = float((timestamp - self.timestamps[lower]) / gap)
        start = self.poses[lower]
        end = self.poses[upper]
        angle_delta = normalize_angle(float(end[2] - start[2]))
        pose: Pose2D = (
            float(start[0] + fraction * (end[0] - start[0])),
            float(start[1] + fraction * (end[1] - start[1])),
            normalize_angle(float(start[2] + fraction * angle_delta)),
        )
        covariance = None
        if self.covariances is not None:
            covariance = (
                (1.0 - fraction) * self.covariances[lower]
                + fraction * self.covariances[upper]
            )
            covariance = 0.5 * (covariance + covariance.T)
        return pose, covariance


def _load_pose_rows(path: str, minimum_fields: int) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    poses: list[Pose2D] = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < minimum_fields:
                raise ValueError(
                    f"{path}:{line_number}: expected at least {minimum_fields} fields"
                )
            try:
                values = [float(value) for value in fields[:8]]
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: malformed numeric field"
                ) from error
            timestamp, px, py, _pz, qx, qy, qz, qw = values
            timestamps.append(timestamp)
            poses.append((px, py, quaternion_yaw(qx, qy, qz, qw)))

    if len(timestamps) < 2:
        raise ValueError(f"{path}: expected at least two pose samples")
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    if np.any(np.diff(timestamp_array) <= 0.0):
        raise ValueError(f"{path}: timestamps must be strictly increasing")
    return timestamp_array, np.asarray(poses, dtype=np.float64)


def load_ground_truth(path: str) -> PlanarPoseStream:
    """Load ``groundtruth.txt`` as a planar ground-truth stream."""
    timestamps, poses = _load_pose_rows(path, minimum_fields=8)
    return PlanarPoseStream(timestamps=timestamps, poses=poses)


def load_wheel_odometry(
    path: str, params: OdometryUncertaintyParams | None = None
) -> PlanarPoseStream:
    """Load recorded ``odom.txt`` and model its cumulative uncertainty."""
    timestamps, poses = _load_pose_rows(path, minimum_fields=14)
    tracker = OdometryUncertaintyTracker(params)
    covariances = np.stack(
        [tracker.step(tuple(float(value) for value in pose)) for pose in poses]
    )
    return PlanarPoseStream(
        timestamps=timestamps,
        poses=poses,
        covariances=covariances,
    )


def timestamped_color_images(color_dir: str) -> list[tuple[float, str]]:
    """Return timestamped PNG images sorted by numeric timestamp."""
    samples: list[tuple[float, str]] = []
    for path in Path(color_dir).glob("*.png"):
        try:
            timestamp = float(path.stem)
        except ValueError:
            continue
        samples.append((timestamp, str(path)))
    samples.sort(key=lambda sample: sample[0])
    if not samples:
        raise ValueError(f"no timestamp-named PNG images found in {color_dir}")
    return samples
