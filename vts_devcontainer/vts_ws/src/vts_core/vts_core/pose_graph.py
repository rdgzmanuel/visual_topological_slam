"""Small, fully specified SE(2) pose-graph optimizer.

For poses ``x_i=(p_i, theta_i)`` and edge measurement ``z_ij``, the residual is
``Log(z_ij^-1 * (x_i^-1 * x_j))`` in the local frame of ``i``.  The objective
uses ordinary Gaussian odometry factors and Huber-robust visual loop factors.
Odometry and loop factors carry separate information matrices stored in
:class:`TopoGraph`.
The first pose is held exactly fixed to remove gauge freedom. Initialization
is the live odometry trajectory. GTSAM Levenberg--Marquardt is the production
backend; a monotonic dense Gauss--Newton/IRLS implementation is retained as a
tested fallback. Both use a 30-iteration limit and ``1e-6`` stopping tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vts_core.motion import normalize_angle
from vts_core.topo_graph import TopoGraph

_HUBER_DELTA = 1.345
_DAMPING = 1e-6
_STEP_TOLERANCE = 1e-6


@dataclass(frozen=True)
class OptimizationResult:
    initial_error: float
    final_error: float
    iterations: int
    converged: bool
    backend: str


def _huber_cost(norm: float) -> float:
    if norm <= _HUBER_DELTA:
        return 0.5 * norm * norm
    return _HUBER_DELTA * (norm - 0.5 * _HUBER_DELTA)


def _huber_weight(norm: float) -> float:
    if norm <= _HUBER_DELTA or norm < 1e-12:
        return 1.0
    return _HUBER_DELTA / norm


def _optimize_numpy(
    graph: TopoGraph, max_iterations: int = 30, backend: str = "numpy"
) -> OptimizationResult:
    """Dense Gauss--Newton fallback used when GTSAM is unavailable."""
    ids = sorted(graph.nodes)
    edges = [edge for edge in graph.edges() if edge in graph.edge_measurements]
    if len(ids) < 2 or not edges:
        return OptimizationResult(0.0, 0.0, 0, True, backend)

    index = {node_id: i for i, node_id in enumerate(ids)}
    state = np.array([graph.nodes[i].pose for i in ids], dtype=np.float64)
    n = len(ids)

    def terms(x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        hessian = np.zeros((3 * n, 3 * n), dtype=np.float64)
        gradient = np.zeros(3 * n, dtype=np.float64)
        cost = 0.0
        for edge in edges:
            a, b = edge
            ia, ib = index[a], index[b]
            z = np.asarray(graph.edge_measurements[edge], dtype=np.float64)
            information = graph.edge_information.get(edge)
            if information is None:
                sigma = max(graph.edge_sigmas.get(edge, 0.5), 0.05)
                information = np.diag(
                    [sigma**-2, sigma**-2, max(0.05, 0.5 * sigma) ** -2]
                )

            theta = x[ia, 2]
            c, s = float(np.cos(theta)), float(np.sin(theta))
            displacement = x[ib, :2] - x[ia, :2]
            predicted_x = c * displacement[0] + s * displacement[1]
            predicted_y = -s * displacement[0] + c * displacement[1]
            residual = np.array(
                [
                    predicted_x - z[0],
                    predicted_y - z[1],
                    normalize_angle((x[ib, 2] - x[ia, 2]) - z[2]),
                ],
                dtype=np.float64,
            )
            whitened_norm = float(np.sqrt(max(residual @ information @ residual, 0.0)))
            is_loop = graph.edge_types.get(edge) == "loop"
            weight = _huber_weight(whitened_norm) if is_loop else 1.0
            robust_information = weight * information
            cost += (
                _huber_cost(whitened_norm)
                if is_loop
                else 0.5 * whitened_norm * whitened_norm
            )

            jacobian_a = np.array(
                [
                    [-c, -s, predicted_y],
                    [s, -c, -predicted_x],
                    [0.0, 0.0, -1.0],
                ]
            )
            jacobian_b = np.array(
                [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
            )
            for jacobian_p, row in ((jacobian_a, ia), (jacobian_b, ib)):
                row_slice = slice(3 * row, 3 * row + 3)
                gradient[row_slice] += (
                    jacobian_p.T @ robust_information @ residual
                )
                for jacobian_q, column in ((jacobian_a, ia), (jacobian_b, ib)):
                    column_slice = slice(3 * column, 3 * column + 3)
                    hessian[row_slice, column_slice] += (
                        jacobian_p.T @ robust_information @ jacobian_q
                    )
        return cost, hessian, gradient

    initial_error, _, _ = terms(state)
    final_error = initial_error
    converged = False
    iterations = 0
    # Exact gauge fixing: remove the first pose's rows/columns from the solve.
    free = slice(3, 3 * n)
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        current_error, hessian, gradient = terms(state)
        reduced_hessian = hessian[free, free] + np.eye(3 * (n - 1)) * _DAMPING
        try:
            reduced_delta = -np.linalg.solve(reduced_hessian, gradient[free])
        except np.linalg.LinAlgError:
            break
        delta = np.zeros(3 * n, dtype=np.float64)
        delta[free] = reduced_delta
        # Backtracking makes the declared robust objective monotonic. Pose
        # updates are committed only when they do not increase that objective.
        step_scale = 1.0
        accepted_state: np.ndarray | None = None
        accepted_error = current_error
        for _ in range(12):
            candidate = state.copy()
            candidate[:, 0] += step_scale * delta[0::3]
            candidate[:, 1] += step_scale * delta[1::3]
            candidate[:, 2] = np.array(
                [
                    normalize_angle(v)
                    for v in candidate[:, 2] + step_scale * delta[2::3]
                ]
            )
            candidate_error, _, _ = terms(candidate)
            if candidate_error <= current_error + 1e-12:
                accepted_state = candidate
                accepted_error = candidate_error
                break
            step_scale *= 0.5
        if accepted_state is None:
            break
        state = accepted_state
        final_error = accepted_error
        if step_scale * float(np.max(np.abs(reduced_delta))) < _STEP_TOLERANCE:
            converged = True
            break

    final_error, _, _ = terms(state)
    for node_id in ids:
        x, y, theta = state[index[node_id]]
        graph.nodes[node_id].pose = (float(x), float(y), normalize_angle(float(theta)))
    return OptimizationResult(
        float(initial_error), float(final_error), iterations, converged, backend
    )


def _optimize_gtsam(
    graph: TopoGraph, max_iterations: int = 30
) -> OptimizationResult:
    """Optimize the same SE(2) factor graph with GTSAM Levenberg--Marquardt."""
    import gtsam

    ids = sorted(graph.nodes)
    edges = [edge for edge in graph.edges() if edge in graph.edge_measurements]
    if len(ids) < 2 or not edges:
        return OptimizationResult(0.0, 0.0, 0, True, "gtsam")

    factor_graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    for node_id in ids:
        initial.insert(node_id, gtsam.Pose2(*graph.nodes[node_id].pose))

    # Gauge fixing. The optimized graph is additionally rigidly transformed
    # below so the first pose is exactly equal to its initialization.
    anchor_id = ids[0]
    anchor_pose = gtsam.Pose2(*graph.nodes[anchor_id].pose)
    anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([1e-9, 1e-9, 1e-9], dtype=np.float64)
    )
    factor_graph.add(gtsam.PriorFactorPose2(anchor_id, anchor_pose, anchor_noise))

    huber = gtsam.noiseModel.mEstimator.Huber.Create(_HUBER_DELTA)
    for a, b in edges:
        information = graph.edge_information.get((a, b))
        if information is None:
            sigma = max(graph.edge_sigmas.get((a, b), 0.5), 0.05)
            information = np.diag(
                [sigma**-2, sigma**-2, max(0.05, 0.5 * sigma) ** -2]
            )
        # Descriptor loop factors deliberately have zero yaw information.
        # A 1e-12 diagonal regularizer is only for GTSAM's matrix
        # factorization and is negligible relative to every real factor.
        regularized = np.asarray(information, dtype=np.float64) + np.eye(3) * 1e-12
        gaussian = gtsam.noiseModel.Gaussian.Information(regularized)
        noise = (
            gtsam.noiseModel.Robust.Create(huber, gaussian)
            if graph.edge_types.get((a, b)) == "loop"
            else gaussian
        )
        measurement = gtsam.Pose2(*graph.edge_measurements[(a, b)])
        factor_graph.add(gtsam.BetweenFactorPose2(a, b, measurement, noise))

    parameters = gtsam.LevenbergMarquardtParams()
    parameters.setMaxIterations(max_iterations)
    parameters.setRelativeErrorTol(_STEP_TOLERANCE)
    parameters.setAbsoluteErrorTol(_STEP_TOLERANCE)
    optimizer = gtsam.LevenbergMarquardtOptimizer(
        factor_graph, initial, parameters
    )
    initial_error = float(factor_graph.error(initial))
    result = optimizer.optimize()
    final_error = float(factor_graph.error(result))
    iterations = int(optimizer.iterations())

    optimized_anchor = result.atPose2(anchor_id)
    correction = anchor_pose.compose(optimized_anchor.inverse())
    accepted = final_error <= initial_error + 1e-9
    if accepted:
        for node_id in ids:
            pose = correction.compose(result.atPose2(node_id))
            graph.nodes[node_id].pose = (
                float(pose.x()),
                float(pose.y()),
                normalize_angle(float(pose.theta())),
            )
    return OptimizationResult(
        initial_error,
        final_error,
        iterations,
        accepted and iterations < max_iterations,
        "gtsam",
    )


def optimize_se2(
    graph: TopoGraph,
    max_iterations: int = 30,
    backend: str = "gtsam",
) -> OptimizationResult:
    """Optimize measured SE(2) factors and mutate node poses in place.

    GTSAM is the production backend. ``numpy`` selects the dependency-free
    reference implementation; if GTSAM is requested but not installed, the
    same reference solver runs and records ``numpy_fallback`` in the result.
    """
    if backend not in {"gtsam", "numpy"}:
        raise ValueError("backend must be 'gtsam' or 'numpy'")
    if backend == "numpy":
        return _optimize_numpy(graph, max_iterations)
    try:
        return _optimize_gtsam(graph, max_iterations)
    except ImportError:
        return _optimize_numpy(graph, max_iterations, "numpy_fallback")
