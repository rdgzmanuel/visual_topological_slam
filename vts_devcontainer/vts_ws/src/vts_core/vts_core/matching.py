"""Matching primitives for visual loop closure and geometric gating.

Metric gating uses a chi-square test on Mahalanobis distance under odometry
and place-extent covariance, so it scales with uncertainty instead of using a
fixed distance threshold. Visual ranking and its MAD-relative acceptance test
live together in :mod:`vts_core.mapper`.
"""

from __future__ import annotations

import numpy as np

# 95th percentile of chi-square with two degrees of freedom.
_CHI2_GATE_2DOF: float = 5.991464547107979


def mahalanobis_gate(
    delta_xy: np.ndarray, covariance_xy: np.ndarray
) -> bool:
    """Chi-square gate (95%, 2 dof) on a position difference.

    Args:
        delta_xy: 2-vector difference between two positions.
        covariance_xy: 2x2 covariance of that difference.

    Returns:
        True if the difference is statistically compatible with zero.
    """
    d2 = mahalanobis_distance_squared(delta_xy, covariance_xy)
    return bool(np.isfinite(d2) and d2 <= _CHI2_GATE_2DOF)


def mahalanobis_distance_squared(
    delta_xy: np.ndarray, covariance_xy: np.ndarray
) -> float:
    """Return the normalized innovation squared for a 2-D displacement."""
    cov: np.ndarray = np.asarray(covariance_xy, dtype=np.float64)
    cov = 0.5 * (cov + cov.T) + np.eye(2) * 1e-6
    try:
        solution = np.linalg.solve(cov, np.asarray(delta_xy, dtype=np.float64))
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.asarray(delta_xy, dtype=np.float64) @ solution)


def gaussian_position_nll(
    delta_xy: np.ndarray, covariance_xy: np.ndarray
) -> float:
    """Gaussian position negative log-likelihood, up to a shared constant.

    Unlike Mahalanobis distance alone, this score includes the covariance
    volume. A highly uncertain candidate therefore becomes uninformative
    instead of automatically looking compatible with every position.
    """
    covariance = np.asarray(covariance_xy, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T) + np.eye(2) * 1e-6
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0.0:
        return float("inf")
    d2 = mahalanobis_distance_squared(delta_xy, covariance)
    return 0.5 * (d2 + float(log_determinant))


def fit_se2(
    points_a: np.ndarray, points_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Least-squares SE(2) fit (Umeyama without scale)."""
    if points_a.shape[0] < 2:
        return None
    centroid_a: np.ndarray = points_a.mean(axis=0)
    centroid_b: np.ndarray = points_b.mean(axis=0)
    centered_a: np.ndarray = points_a - centroid_a
    centered_b: np.ndarray = points_b - centroid_b

    h: np.ndarray = centered_a.T @ centered_b
    u, _, vt = np.linalg.svd(h)
    d: float = float(np.sign(np.linalg.det(vt.T @ u.T)))
    correction: np.ndarray = np.diag([1.0, d])
    rotation: np.ndarray = vt.T @ correction @ u.T
    translation: np.ndarray = centroid_b - rotation @ centroid_a
    return rotation, translation


def median_spacing(points: np.ndarray) -> float:
    """Median nearest-neighbor distance of a point set (data-driven scale).

    Args:
        points: (N, 2) positions.

    Returns:
        Median NN distance, or 1.0 for degenerate inputs.
    """
    n: int = points.shape[0]
    if n < 2:
        return 1.0
    deltas: np.ndarray = points[:, None, :] - points[None, :, :]
    distances: np.ndarray = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(distances, np.inf)
    nn: np.ndarray = distances.min(axis=1)
    return float(np.median(nn))
