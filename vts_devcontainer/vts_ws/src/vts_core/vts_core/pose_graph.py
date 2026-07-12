"""Lightweight SE(2) pose-graph optimization (pure NumPy, no gtsam).

A single global Gauss-Newton solve with a Huber robust kernel, used at
end-of-run to correct accumulated odometry drift. It consumes the relative
SE(2) measurements that the mapper stores per edge at creation time
(:meth:`TopoGraph.set_edge_measurement`) together with their translational
sigmas, and rewrites every node pose in a single consistent, drift-corrected
frame anchored at the first node.

Why this exists / why it is small:
- The earlier design used gtsam, an aarch64-fragile dependency that the rest
  of the (single-run) pipeline no longer needs. The optimization problem here
  is tiny (a few dozen nodes), so a dense Gauss-Newton step solved with
  ``numpy.linalg.solve`` is more than enough and adds no dependency.
- The Huber kernel down-weights inconsistent loop closures (the residual
  false closures that perceptual aliasing leaves behind) instead of letting
  them deform the whole map.
"""

from __future__ import annotations

import numpy as np

from vts_core.motion import normalize_angle
from vts_core.topo_graph import TopoGraph

_HUBER_K: float = 1.345  # 95%-efficiency Huber constant
_MIN_SIGMA: float = 0.05
_DAMPING: float = 1e-6  # Levenberg damping for a well-posed normal matrix


def _huber_weight(whitened_norm: float) -> float:
    """Multiplicative weight of the Huber M-estimator at a residual norm."""
    if whitened_norm <= _HUBER_K or whitened_norm < 1e-12:
        return 1.0
    return _HUBER_K / whitened_norm


def optimize_se2(graph: TopoGraph, iterations: int = 30) -> tuple[float, float]:
    """Globally optimize node poses from stored relative measurements.

    Mutates ``graph`` node poses in place.

    Args:
        graph: Map whose edges carry ``edge_measurements`` (relative SE(2),
            oriented min-id -> max-id) and ``edge_sigmas``.
        iterations: Maximum Gauss-Newton iterations.

    Returns:
        (initial_error, final_error): the robustified squared error before and
        after optimization, for logging. ``(0.0, 0.0)`` if there is nothing to
        optimize.
    """
    ids: list[int] = sorted(graph.nodes)
    edges: list[tuple[int, int]] = [
        (a, b)
        for (a, b) in graph.edges()
        if (a, b) in graph.edge_measurements
    ]
    if len(ids) < 3 or len(edges) < 2:
        return (0.0, 0.0)

    index: dict[int, int] = {nid: k for k, nid in enumerate(ids)}
    n: int = len(ids)
    state: np.ndarray = np.array(
        [graph.nodes[i].pose for i in ids], dtype=np.float64
    )

    def edge_terms(
        x: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Return (robust error, normal matrix H, gradient g) at state x."""
        size: int = 3 * n
        h_mat: np.ndarray = np.zeros((size, size), dtype=np.float64)
        grad: np.ndarray = np.zeros(size, dtype=np.float64)
        error: float = 0.0
        for a, b in edges:
            ia, ib = index[a], index[b]
            z: np.ndarray = np.array(
                graph.edge_measurements[(a, b)], dtype=np.float64
            )
            sigma: float = max(graph.edge_sigmas.get((a, b), 0.5), _MIN_SIGMA)
            info: np.ndarray = np.diag(
                [1.0 / sigma**2, 1.0 / sigma**2, 1.0 / max(0.05, sigma * 0.5) ** 2]
            )

            ti: float = x[ia, 2]
            c, s = np.cos(ti), np.sin(ti)
            dp: np.ndarray = x[ib, :2] - x[ia, :2]
            pred_x: float = c * dp[0] + s * dp[1]
            pred_y: float = -s * dp[0] + c * dp[1]
            residual: np.ndarray = np.array(
                [
                    pred_x - z[0],
                    pred_y - z[1],
                    normalize_angle((x[ib, 2] - x[ia, 2]) - z[2]),
                ]
            )

            whitened: float = float(
                np.sqrt(residual @ info @ residual)
            )
            weight: float = _huber_weight(whitened)
            w_info: np.ndarray = weight * info
            error += weight * float(residual @ info @ residual)

            j_i: np.ndarray = np.array(
                [[-c, -s, pred_y], [s, -c, -pred_x], [0.0, 0.0, -1.0]]
            )
            j_j: np.ndarray = np.array(
                [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
            )

            blocks = ((j_i, ia), (j_j, ib))
            for (jp, rp) in blocks:
                grad[3 * rp : 3 * rp + 3] += jp.T @ w_info @ residual
                for (jq, cq) in blocks:
                    h_mat[3 * rp : 3 * rp + 3, 3 * cq : 3 * cq + 3] += (
                        jp.T @ w_info @ jq
                    )
        return error, h_mat, grad

    initial_error, _, _ = edge_terms(state)

    # Gauge fix: pin the first node by heavily weighting its block.
    anchor_slice: slice = slice(0, 3)
    final_error: float = initial_error
    for _ in range(iterations):
        final_error, h_mat, grad = edge_terms(state)
        h_mat[anchor_slice, anchor_slice] += np.eye(3) * 1e12
        h_mat += np.eye(3 * n) * _DAMPING
        try:
            delta: np.ndarray = -np.linalg.solve(h_mat, grad)
        except np.linalg.LinAlgError:
            break
        state[:, 0] += delta[0::3]
        state[:, 1] += delta[1::3]
        state[:, 2] = (state[:, 2] + delta[2::3] + np.pi) % (2.0 * np.pi) - np.pi
        if float(np.max(np.abs(delta))) < 1e-6:
            break

    final_error, _, _ = edge_terms(state)
    for nid in ids:
        x, y, theta = state[index[nid]]
        graph.nodes[nid].pose = (float(x), float(y), normalize_angle(float(theta)))
    return (float(initial_error), float(final_error))
