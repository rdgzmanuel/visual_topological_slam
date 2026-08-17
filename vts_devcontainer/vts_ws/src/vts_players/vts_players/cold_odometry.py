"""Pure-Python reader and synchronizer for COLD `odom.tdf` files."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vts_core.motion import (
    OdometryUncertaintyParams,
    OdometryUncertaintyTracker,
    Pose2D,
    normalize_angle,
)


@dataclass(frozen=True)
class ColdOdometry:
    """Timestamped recorded poses and modeled cumulative uncertainty budgets."""

    timestamps: np.ndarray
    poses: np.ndarray
    covariances: np.ndarray

    def at(self, timestamp: float, max_gap_s: float = 1.0) -> tuple[Pose2D, np.ndarray]:
        """Linearly interpolate pose and covariance at a camera timestamp."""
        if timestamp < self.timestamps[0] or timestamp > self.timestamps[-1]:
            raise ValueError(f"timestamp {timestamp:.6f} is outside odometry coverage")

        upper = int(np.searchsorted(self.timestamps, timestamp, side="left"))
        if upper < len(self.timestamps) and np.isclose(
            self.timestamps[upper], timestamp, atol=1e-7, rtol=0.0
        ):
            pose = self.poses[upper]
            return tuple(float(v) for v in pose), self.covariances[upper].copy()
        if upper == 0 or upper == len(self.timestamps):
            raise ValueError(f"cannot bracket odometry timestamp {timestamp:.6f}")

        lower = upper - 1
        gap = float(self.timestamps[upper] - self.timestamps[lower])
        if gap <= 0.0 or gap > max_gap_s:
            raise ValueError(
                f"odometry gap {gap:.3f}s around timestamp {timestamp:.6f} "
                f"exceeds the {max_gap_s:.3f}s limit"
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
        covariance = (
            (1.0 - fraction) * self.covariances[lower]
            + fraction * self.covariances[upper]
        )
        covariance = 0.5 * (covariance + covariance.T)
        return pose, covariance


def load_cold_odometry(
    path: str, params: OdometryUncertaintyParams | None = None
) -> ColdOdometry:
    """Read COLD odometry and accumulate uncertainty over recorded poses.

    Each TDF row stores seconds and microseconds at columns 3--4 and
    `x, y, theta` at columns 8, 9 and 11 respectively.
    """
    timestamps: list[float] = []
    poses: list[Pose2D] = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 12:
                raise ValueError(f"{path}:{line_number}: expected at least 12 fields")
            try:
                timestamp = float(fields[3]) + 1e-6 * float(fields[4])
                pose = (float(fields[8]), float(fields[9]), float(fields[11]))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: malformed numeric field") from error
            timestamps.append(timestamp)
            poses.append(pose)

    if len(timestamps) < 2:
        raise ValueError(f"{path}: expected at least two odometry samples")
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    if np.any(np.diff(timestamp_array) <= 0.0):
        raise ValueError(f"{path}: odometry timestamps must be strictly increasing")

    tracker = OdometryUncertaintyTracker(params)
    covariance_array = np.stack([tracker.step(pose) for pose in poses])
    return ColdOdometry(
        timestamps=timestamp_array,
        poses=np.asarray(poses, dtype=np.float64),
        covariances=covariance_array,
    )
