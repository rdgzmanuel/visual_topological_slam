"""Evaluation metrics for topological maps and language retrieval.

These mirror the *classes* of metrics reported by PRISM-TopoMap (Muravyev et
al., RA-L 2025) — graph size/compactness, mapping quality against ground
truth, loop-closure correctness, and computational cost — so your tables can
be put side-by-side with theirs. Note the honest caveat for the paper:
PRISM-TopoMap consumes point clouds + multi-camera input and was evaluated
in Habitat and on a Husky; running *your* method on *their* data (your
stated plan) yields comparable numbers on shared metrics, but is not a
same-input head-to-head and should be described as such.

Conventions:
- Ground truth is a timestamped trajectory [(t, x, y, theta), ...] (for COLD,
  parsed from filenames).
- Spatial tolerances are derived from the data (median inter-node spacing)
  unless the caller overrides them explicitly for protocol compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vts_core.matching import median_spacing
from vts_core.topo_graph import TopoGraph


@dataclass
class TrajectoryMetrics:
    """Accuracy of an estimated trajectory against ground truth."""

    ate_rmse: float
    ate_median: float
    final_drift: float


@dataclass
class GraphMetrics:
    """Structural and quality metrics of a topological map."""

    n_nodes: int
    n_edges: int
    coverage: float
    node_placement_rmse: float
    false_merge_rate: float
    spatial_tolerance: float


def trajectory_metrics(
    estimated: np.ndarray, ground_truth: np.ndarray
) -> TrajectoryMetrics:
    """ATE-style metrics between aligned, same-length (N, 2) trajectories.

    Args:
        estimated: (N, 2) estimated positions (e.g. simulated odometry).
        ground_truth: (N, 2) ground-truth positions.

    Returns:
        TrajectoryMetrics with RMSE, median error and final drift.
    """
    if estimated.shape != ground_truth.shape:
        raise ValueError("Trajectories must have identical shapes")
    errors: np.ndarray = np.linalg.norm(estimated - ground_truth, axis=1)
    return TrajectoryMetrics(
        ate_rmse=float(np.sqrt(np.mean(errors**2))),
        ate_median=float(np.median(errors)),
        final_drift=float(errors[-1]),
    )


def graph_metrics(
    graph: TopoGraph,
    ground_truth_xy: np.ndarray,
    node_gt_xy: dict[int, np.ndarray] | None = None,
    spatial_tolerance: float | None = None,
) -> GraphMetrics:
    """Quality metrics of a topological map against a ground-truth trajectory.

    Args:
        graph: The map to evaluate. Node poses may live in a drifting
            odometry frame; for placement metrics provide ``node_gt_xy``.
        ground_truth_xy: (N, 2) ground-truth trajectory positions.
        node_gt_xy: Optional mapping node_id -> ground-truth (x, y) of the
            frame that created the node (lets placement error be measured in
            the GT frame, immune to odometric drift of the whole map).
        spatial_tolerance: Radius for coverage / merge checks. Default:
            median inter-node spacing (data-driven).

    Returns:
        GraphMetrics. ``coverage`` is the fraction of trajectory samples
        within tolerance of some node; ``false_merge_rate`` is the fraction
        of edges whose endpoints' GT positions are farther apart than
        3x tolerance (perceptual-aliasing errors); placement RMSE compares
        node poses against the nearest GT trajectory point (or
        ``node_gt_xy`` when given).
    """
    positions: np.ndarray = (
        np.array([node_gt_xy[i] for i in sorted(graph.nodes)], dtype=np.float64)
        if node_gt_xy is not None
        else graph.positions()
    )
    tolerance: float = (
        spatial_tolerance
        if spatial_tolerance is not None
        else median_spacing(positions)
    )

    # Coverage: every GT sample should be near some node.
    if positions.size and ground_truth_xy.size:
        deltas: np.ndarray = (
            ground_truth_xy[:, None, :] - positions[None, :, :]
        )
        distances: np.ndarray = np.linalg.norm(deltas, axis=2)
        nearest: np.ndarray = distances.min(axis=1)
        coverage: float = float(np.mean(nearest <= tolerance))
        placement: float = float(
            np.sqrt(np.mean(distances.min(axis=0) ** 2))
        )
    else:
        coverage = 0.0
        placement = float("nan")

    # False merges / wrong edges in the GT frame.
    ids_sorted: list[int] = sorted(graph.nodes)
    index_of: dict[int, int] = {nid: k for k, nid in enumerate(ids_sorted)}
    edge_count: int = 0
    bad_edges: int = 0
    for id_a, id_b in graph.edges():
        edge_count += 1
        pa: np.ndarray = positions[index_of[id_a]]
        pb: np.ndarray = positions[index_of[id_b]]
        if float(np.linalg.norm(pa - pb)) > 3.0 * tolerance:
            bad_edges += 1
    false_merge_rate: float = bad_edges / edge_count if edge_count else 0.0

    return GraphMetrics(
        n_nodes=len(graph.nodes),
        n_edges=edge_count,
        coverage=coverage,
        node_placement_rmse=placement,
        false_merge_rate=false_merge_rate,
        spatial_tolerance=tolerance,
    )


def retrieval_metrics(
    ranked_labels: list[list[str]], true_labels: list[str], k: int = 3
) -> dict[str, float]:
    """Recall@1 / Recall@k / MRR for language-to-node retrieval.

    Args:
        ranked_labels: For each query, the room labels of the returned nodes
            in rank order.
        true_labels: The correct room label per query.
        k: Cutoff for Recall@k.

    Returns:
        Dict with recall_at_1, recall_at_k, mrr.
    """
    if len(ranked_labels) != len(true_labels):
        raise ValueError("ranked_labels and true_labels length mismatch")
    if not true_labels:
        return {"recall_at_1": 0.0, f"recall_at_{k}": 0.0, "mrr": 0.0}

    hits_1: int = 0
    hits_k: int = 0
    reciprocal_ranks: list[float] = []
    for ranks, truth in zip(ranked_labels, true_labels):
        if ranks and ranks[0] == truth:
            hits_1 += 1
        if truth in ranks[:k]:
            hits_k += 1
        rank_position: int = next(
            (r + 1 for r, label in enumerate(ranks) if label == truth), 0
        )
        reciprocal_ranks.append(1.0 / rank_position if rank_position else 0.0)

    n: int = len(true_labels)
    return {
        "recall_at_1": hits_1 / n,
        f"recall_at_{k}": hits_k / n,
        "mrr": float(np.mean(reciprocal_ranks)),
    }
