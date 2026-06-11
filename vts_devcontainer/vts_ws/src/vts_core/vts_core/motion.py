"""Motion utilities and odometry simulation.

The odometry simulator implements the standard probabilistic odometry motion
model (Thrun, Burgard & Fox, *Probabilistic Robotics*, ch. 5.4): each relative
motion between consecutive ground-truth poses is decomposed into an initial
rotation, a translation, and a final rotation, each perturbed with zero-mean
Gaussian noise whose variance scales with the magnitudes of the motion
components. Integrating the perturbed increments yields a drifting trajectory
that is statistically representative of real wheel odometry — unlike additive
white noise on absolute coordinates, which has no drift and would be rejected
by any reviewer familiar with odometric error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Pose2D = tuple[float, float, float]


def normalize_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def relative_motion(prev: Pose2D, curr: Pose2D) -> tuple[float, float, float]:
    """Decompose the motion between two poses into (rot1, trans, rot2).

    Args:
        prev: Previous pose (x, y, theta).
        curr: Current pose (x, y, theta).

    Returns:
        rot1: Initial heading change toward the new position.
        trans: Translated distance.
        rot2: Remaining rotation to reach the new heading.
    """
    dx: float = curr[0] - prev[0]
    dy: float = curr[1] - prev[1]
    trans: float = float(np.hypot(dx, dy))

    if trans < 1e-6:
        # Pure rotation: attribute everything to rot2 (convention).
        rot1: float = 0.0
        rot2: float = normalize_angle(curr[2] - prev[2])
    else:
        rot1 = normalize_angle(float(np.arctan2(dy, dx)) - prev[2])
        rot2 = normalize_angle(curr[2] - prev[2] - rot1)
    return rot1, trans, rot2


def compose_motion(pose: Pose2D, rot1: float, trans: float, rot2: float) -> Pose2D:
    """Apply a (rot1, trans, rot2) increment to a pose."""
    theta_mid: float = pose[2] + rot1
    x: float = pose[0] + trans * float(np.cos(theta_mid))
    y: float = pose[1] + trans * float(np.sin(theta_mid))
    theta: float = normalize_angle(theta_mid + rot2)
    return (x, y, theta)


@dataclass
class OdometryNoiseParams:
    """Noise coefficients of the odometry motion model.

    alpha1: rotational noise from rotation (rad^2 / rad^2).
    alpha2: rotational noise from translation (rad^2 / m^2).
    alpha3: translational noise from translation (m^2 / m^2).
    alpha4: translational noise from rotation (m^2 / rad^2).

    Defaults are typical values for an indoor differential-drive platform.
    They are configuration, not hidden constants: expose them as ROS
    parameters so the simulated drift severity is an explicit experimental
    variable in the paper.
    """

    alpha1: float = 0.05
    alpha2: float = 0.01
    alpha3: float = 0.02
    alpha4: float = 0.005


class OdometrySimulator:
    """Generates realistic odometry from a stream of ground-truth poses.

    Feed ground-truth poses (e.g. parsed from COLD image filenames) in
    chronological order; the simulator perturbs each relative motion and
    integrates it, also propagating a diagonal position covariance that
    downstream modules use for probabilistic gating.
    """

    def __init__(
        self,
        params: OdometryNoiseParams | None = None,
        seed: int | None = 17,
    ) -> None:
        self._params: OdometryNoiseParams = params or OdometryNoiseParams()
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._gt_prev: Pose2D | None = None
        self._pose: Pose2D = (0.0, 0.0, 0.0)
        self._covariance: np.ndarray = np.zeros((3, 3), dtype=np.float64)

    @property
    def pose(self) -> Pose2D:
        """Current simulated odometry pose."""
        return self._pose

    @property
    def covariance(self) -> np.ndarray:
        """Current 3x3 pose covariance (x, y, theta)."""
        return self._covariance.copy()

    def step(self, ground_truth: Pose2D) -> tuple[Pose2D, np.ndarray]:
        """Advance the simulator with a new ground-truth pose.

        Args:
            ground_truth: The true pose at the current time step.

        Returns:
            The new (drifting) odometry pose and its 3x3 covariance.
        """
        if self._gt_prev is None:
            self._gt_prev = ground_truth
            self._pose = ground_truth
            return self._pose, self._covariance.copy()

        rot1, trans, rot2 = relative_motion(self._gt_prev, ground_truth)
        p: OdometryNoiseParams = self._params

        std_rot1: float = float(
            np.sqrt(p.alpha1 * rot1**2 + p.alpha2 * trans**2)
        )
        std_trans: float = float(
            np.sqrt(p.alpha3 * trans**2 + p.alpha4 * (rot1**2 + rot2**2))
        )
        std_rot2: float = float(
            np.sqrt(p.alpha1 * rot2**2 + p.alpha2 * trans**2)
        )

        noisy_rot1: float = rot1 + float(self._rng.normal(0.0, std_rot1))
        noisy_trans: float = trans + float(self._rng.normal(0.0, std_trans))
        noisy_rot2: float = rot2 + float(self._rng.normal(0.0, std_rot2))

        self._pose = compose_motion(self._pose, noisy_rot1, noisy_trans, noisy_rot2)
        self._gt_prev = ground_truth

        # First-order covariance propagation with a diagonal increment noise.
        theta: float = self._pose[2]
        jacobian: np.ndarray = np.array(
            [
                [1.0, 0.0, -noisy_trans * np.sin(theta)],
                [0.0, 1.0, noisy_trans * np.cos(theta)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        increment_cov: np.ndarray = np.diag(
            [std_trans**2 + 1e-12, std_trans**2 + 1e-12, std_rot1**2 + std_rot2**2 + 1e-12]
        )
        self._covariance = (
            jacobian @ self._covariance @ jacobian.T + increment_cov
        )

        return self._pose, self._covariance.copy()
