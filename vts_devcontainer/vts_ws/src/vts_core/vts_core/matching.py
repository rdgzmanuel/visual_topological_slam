"""Matching primitives shared by loop closure and multi-map alignment.

Design principle: replace absolute thresholds with *relative* and
*probabilistic* criteria.

- Visual association uses mutual nearest neighbors plus Lowe's ratio test on
  cosine distances; both are relative criteria with standard, citable
  constants rather than per-dataset magic numbers.
- Metric gating uses the chi-square test on the Mahalanobis distance under
  the odometry covariance, so the gate widens automatically as uncertainty
  grows instead of using a fixed metres threshold.
- Cross-map geometric consistency is enforced with an SE(2) RANSAC whose
  inlier tolerance is derived from the data (median node spacing).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

_LOWE_RATIO: float = 0.8
_CHI2_CONFIDENCE: float = 0.95
_CHI2_GATE_2DOF: float = float(chi2.ppf(_CHI2_CONFIDENCE, df=2))


def mutual_nearest_neighbors(
    features_a: np.ndarray, features_b: np.ndarray
) -> list[tuple[int, int, float]]:
    """Mutual-NN matches between two sets of L2-normalized descriptors.

    A pair (i, j) is kept if j is i's nearest neighbor in B, i is j's nearest
    neighbor in A, and i's best similarity passes Lowe's ratio test against
    its second-best (applied in cosine-distance space).

    Args:
        features_a: (Na, d) normalized descriptors.
        features_b: (Nb, d) normalized descriptors.

    Returns:
        List of (index_a, index_b, cosine_similarity) tuples.
    """
    if features_a.size == 0 or features_b.size == 0:
        return []

    similarity: np.ndarray = features_a @ features_b.T  # (Na, Nb)
    best_b_for_a: np.ndarray = np.argmax(similarity, axis=1)
    best_a_for_b: np.ndarray = np.argmax(similarity, axis=0)

    matches: list[tuple[int, int, float]] = []
    for i in range(similarity.shape[0]):
        j: int = int(best_b_for_a[i])
        if int(best_a_for_b[j]) != i:
            continue

        row: np.ndarray = similarity[i]
        if row.shape[0] >= 2:
            sorted_row: np.ndarray = np.sort(row)[::-1]
            best_dist: float = 1.0 - float(sorted_row[0])
            second_dist: float = 1.0 - float(sorted_row[1])
            if second_dist > 1e-12 and best_dist / second_dist > _LOWE_RATIO:
                continue
        matches.append((i, j, float(row[j])))
    return matches


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
    cov: np.ndarray = covariance_xy + np.eye(2) * 1e-6
    try:
        inv: np.ndarray = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return False
    d2: float = float(delta_xy @ inv @ delta_xy)
    return d2 <= _CHI2_GATE_2DOF


def estimate_se2_ransac(
    points_a: np.ndarray,
    points_b: np.ndarray,
    inlier_tolerance: float,
    iterations: int = 500,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimate a rigid SE(2) transform mapping points_a onto points_b.

    Args:
        points_a: (N, 2) source positions.
        points_b: (N, 2) corresponding target positions.
        inlier_tolerance: Residual (m) under which a pair counts as inlier;
            derive it from the data (e.g. median inter-node spacing), not a
            constant.
        iterations: RANSAC iterations.
        seed: RNG seed for reproducibility.

    Returns:
        (2x2 rotation, 2-translation) of the best model and refit on its
        inliers, or None if no model with >= 3 inliers exists.
    """
    n: int = points_a.shape[0]
    if n < 2:
        return None

    rng: np.random.Generator = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None

    for _ in range(iterations):
        idx: np.ndarray = rng.choice(n, size=2, replace=False)
        model: tuple[np.ndarray, np.ndarray] | None = _fit_se2(
            points_a[idx], points_b[idx]
        )
        if model is None:
            continue
        rotation, translation = model
        residuals: np.ndarray = np.linalg.norm(
            (points_a @ rotation.T + translation) - points_b, axis=1
        )
        inliers: np.ndarray = residuals < inlier_tolerance
        if best_inliers is None or int(inliers.sum()) > int(best_inliers.sum()):
            best_inliers = inliers

    if best_inliers is None or int(best_inliers.sum()) < 3:
        return None

    refit: tuple[np.ndarray, np.ndarray] | None = _fit_se2(
        points_a[best_inliers], points_b[best_inliers]
    )
    return refit


def _fit_se2(
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
