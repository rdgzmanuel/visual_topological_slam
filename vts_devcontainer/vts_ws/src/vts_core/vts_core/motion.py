"""SE(2) motion helpers and wheel-odometry uncertainty propagation.

The mapper consumes recorded odometry poses unchanged. COLD does not provide
pose covariances, so :class:`OdometryUncertaintyTracker` propagates a standard
motion-dependent uncertainty model over those measurements. Unlike the old
simulator, this module never samples noise and never modifies the trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Pose2D = tuple[float, float, float]


def normalize_angle(angle: float) -> float:
    """Wrap an angle to `[-pi, pi)`."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class OdometryUncertaintyParams:
    """Motion-dependent wheel-odometry variance coefficients.

    `alpha1` and `alpha2` model rotational variance caused by rotation and
    translation. `alpha3` and `alpha4` model translational variance caused
    by translation and rotation. One fixed set must be used for every run.
    """

    alpha1: float = 0.025
    alpha2: float = 0.005
    alpha3: float = 0.01
    alpha4: float = 0.0025

    def __post_init__(self) -> None:
        if min(self.alpha1, self.alpha2, self.alpha3, self.alpha4) < 0.0:
            raise ValueError("odometry uncertainty coefficients must be nonnegative")


class OdometryUncertaintyTracker:
    """Accumulate an additive uncertainty budget for recorded SE(2) motion.

    The returned matrix is monotonic: subtracting two snapshots gives the
    uncertainty accumulated only over that interval. This is the quantity
    required by relative odometry factors and revisit gating. No noise is
    sampled and the recorded trajectory is never modified.
    """

    def __init__(
        self, params: OdometryUncertaintyParams | None = None
    ) -> None:
        self._params = params or OdometryUncertaintyParams()
        self._previous_pose: Pose2D | None = None
        self._covariance = np.zeros((3, 3), dtype=np.float64)

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance.copy()

    def step(self, measured_pose: Pose2D) -> np.ndarray:
        """Consume a recorded pose and return its propagated 3x3 covariance."""
        if self._previous_pose is None:
            self._previous_pose = measured_pose
            return self.covariance

        dx = measured_pose[0] - self._previous_pose[0]
        dy = measured_pose[1] - self._previous_pose[1]
        translation = float(np.hypot(dx, dy))
        rotation = normalize_angle(measured_pose[2] - self._previous_pose[2])
        params = self._params
        variance_rotation = (
            params.alpha1 * rotation**2 + params.alpha2 * translation**2
        )
        variance_translation = (
            params.alpha3 * translation**2
            + params.alpha4 * rotation**2
        )
        increment_covariance = np.diag(
            [variance_translation, variance_translation, variance_rotation]
        )
        self._covariance += increment_covariance
        self._covariance = 0.5 * (self._covariance + self._covariance.T)
        self._previous_pose = measured_pose
        return self.covariance
