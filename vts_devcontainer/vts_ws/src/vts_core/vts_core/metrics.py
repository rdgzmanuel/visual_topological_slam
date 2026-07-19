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

from vts_core.matching import _fit_se2, median_spacing
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
    coverage_1m: float
    coverage_2m: float
    node_placement_rmse: float
    false_merge_rate: float
    spatial_tolerance: float
    # Structural diagnostics (no ground truth needed).
    max_degree: int
    mean_degree: float
    n_components: int
    n_loop_closures: int
    # Edge-length diagnostics (ground-truth frame when node_gt is given).
    median_edge_length: float
    max_edge_length: float


def _connected_components(graph: TopoGraph) -> int:
    """Number of connected components of the undirected graph."""
    seen: set[int] = set()
    components: int = 0
    for start in graph.nodes:
        if start in seen:
            continue
        components += 1
        stack: list[int] = [start]
        while stack:
            node_id: int = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            stack.extend(graph.adjacency.get(node_id, set()) - seen)
    return components


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
        # Fixed radii: comparable ACROSS runs. The adaptive-tolerance
        # coverage is NOT (denser graphs shrink the tolerance, deflating
        # the number even when the map improved).
        coverage_1m: float = float(np.mean(nearest <= 1.0))
        coverage_2m: float = float(np.mean(nearest <= 2.0))
    else:
        coverage = 0.0
        coverage_1m = 0.0
        coverage_2m = 0.0

    # Placement: internal map distortion. With node_gt available, compare the
    # graph's own (odometry-frame, post-optimization) node layout against the
    # GT layout after a best-fit SE(2) alignment. Comparing node_gt positions
    # to the trajectory they were sampled from would be circular and always
    # return 0, so the map-frame layout must be the source of the fit.
    placement: float = float("nan")
    if node_gt_xy is not None and len(graph.nodes) >= 3:
        ids_for_fit: list[int] = sorted(graph.nodes)
        source: np.ndarray = graph.positions()
        target: np.ndarray = np.array(
            [node_gt_xy[i] for i in ids_for_fit], dtype=np.float64
        )
        fit: tuple[np.ndarray, np.ndarray] | None = _fit_se2(source, target)
        if fit is not None:
            rotation, translation = fit
            residuals: np.ndarray = np.linalg.norm(
                (source @ rotation.T + translation) - target, axis=1
            )
            placement = float(np.sqrt(np.mean(residuals**2)))
    elif positions.size and ground_truth_xy.size:
        # No node GT: fall back to nearest-trajectory distance — only
        # meaningful if the graph poses live in the GT frame.
        placement = float(np.sqrt(np.mean(distances.min(axis=0) ** 2)))

    # False merges: perceptual-aliasing edges, defined as edges where the
    # ground truth distance between the endpoints far exceeds the distance
    # the map itself believes (its own pose difference). The reference must
    # be the map's own belief, not a global density statistic, which would
    # wrongly flag legitimately long corridor edges.
    ids_sorted: list[int] = sorted(graph.nodes)
    index_of: dict[int, int] = {nid: k for k, nid in enumerate(ids_sorted)}
    map_positions: np.ndarray = graph.positions()
    edge_count: int = 0
    bad_edges: int = 0
    edge_lengths: list[float] = []
    for id_a, id_b in graph.edges():
        edge_count += 1
        gt_dist: float = float(
            np.linalg.norm(positions[index_of[id_a]] - positions[index_of[id_b]])
        )
        map_dist: float = float(
            np.linalg.norm(
                map_positions[index_of[id_a]] - map_positions[index_of[id_b]]
            )
        )
        edge_lengths.append(gt_dist)
        if gt_dist > map_dist + 3.0 * tolerance:
            bad_edges += 1
    false_merge_rate: float = bad_edges / edge_count if edge_count else 0.0

    # Structural diagnostics. Hubs (very high degree) flagged the
    # descriptor-pollution failure mode; the loop-closure count is the circuit
    # rank E - V + C (independent cycles), i.e. how many loops the map closed.
    degrees: list[int] = [len(graph.adjacency.get(i, set())) for i in graph.nodes]
    n_components: int = _connected_components(graph)
    n_loop_closures: int = max(edge_count - len(graph.nodes) + n_components, 0)

    return GraphMetrics(
        n_nodes=len(graph.nodes),
        n_edges=edge_count,
        coverage=coverage,
        coverage_1m=coverage_1m,
        coverage_2m=coverage_2m,
        node_placement_rmse=placement,
        false_merge_rate=false_merge_rate,
        spatial_tolerance=tolerance,
        max_degree=max(degrees, default=0),
        mean_degree=float(np.mean(degrees)) if degrees else 0.0,
        n_components=n_components,
        n_loop_closures=n_loop_closures,
        median_edge_length=float(np.median(edge_lengths)) if edge_lengths else 0.0,
        max_edge_length=float(np.max(edge_lengths)) if edge_lengths else 0.0,
    )


def descriptor_separation(graph: TopoGraph) -> dict[str, float]:
    """How well node descriptors separate rooms (place-recognition sanity).

    This is the diagnostic that distinguishes a weak encoder from a mapping
    bug: if same-room similarity is well above different-room similarity, the
    descriptors are fine and any aliasing comes from the gating / fusion logic,
    not the encoder. Requires room labels on the nodes.

    Returns:
        Dict with intra/inter-room mean cosine similarity, their separation,
        the fraction of nodes whose nearest other node is the same room, and
        the labelled-node / room counts. Empty if fewer than two labelled
        nodes. ``separation`` near zero (or negative) is a red flag.
    """
    ids: list[int] = [
        i for i in sorted(graph.nodes) if graph.nodes[i].room_label
    ]
    if len(ids) < 2:
        return {}
    labels: list[str] = [str(graph.nodes[i].room_label) for i in ids]
    features: np.ndarray = np.stack(
        [graph.nodes[i].visual_features for i in ids]
    ).astype(np.float64)
    features = features / np.maximum(
        np.linalg.norm(features, axis=1, keepdims=True), 1e-12
    )
    similarity: np.ndarray = features @ features.T

    intra: list[float] = []
    inter: list[float] = []
    nn_same: int = 0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            (intra if labels[a] == labels[b] else inter).append(
                float(similarity[a, b])
            )
        row: np.ndarray = similarity[a].copy()
        row[a] = -np.inf
        if labels[int(np.argmax(row))] == labels[a]:
            nn_same += 1

    intra_mean: float = float(np.mean(intra)) if intra else float("nan")
    inter_mean: float = float(np.mean(inter)) if inter else float("nan")
    return {
        "intra_room_mean": round(intra_mean, 4),
        "inter_room_mean": round(inter_mean, 4),
        "separation": round(intra_mean - inter_mean, 4),
        "nn_same_room_rate": round(nn_same / len(ids), 4),
        "n_labeled_nodes": len(ids),
        "n_rooms": len(set(labels)),
    }


def map_footprint(graph: TopoGraph) -> dict[str, float]:
    """Map size for the PRISM Table V comparison ("Map size, MB").

    Reports both the full in-memory footprint (descriptors + stored RGB views)
    and the serialized size of a descriptors-only map (no images). VTS keeps RGB
    views for the downstream language module, so its full map is necessarily
    larger than PRISM's 0.4 MB (which stores descriptors + 2D scans, no raw
    RGB); the descriptors-only number is the fair like-for-like analogue.

    Returns:
        Dict of byte counts: ``descriptor_bytes``, ``view_bytes``,
        ``in_memory_bytes`` (their sum) and ``descriptors_only_pickle_bytes``.
    """
    import pickle

    descriptor_bytes: int = 0
    view_bytes: int = 0
    descriptors_only: dict[int, np.ndarray] = {}
    for node_id, node in graph.nodes.items():
        features: np.ndarray = np.asarray(node.visual_features)
        descriptor_bytes += int(features.nbytes)
        descriptors_only[node_id] = features.astype(np.float32)
        for view in node.views:
            if view is not None:
                view_bytes += int(np.asarray(view).nbytes)
    payload: dict[str, object] = {
        "descriptors": descriptors_only,
        "adjacency": graph.adjacency,
        "edges": graph.edges(),
        "edge_measurements": graph.edge_measurements,
    }
    serialized: int = len(
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    )
    return {
        "n_nodes": float(len(graph.nodes)),
        "descriptor_bytes": float(descriptor_bytes),
        "view_bytes": float(view_bytes),
        "in_memory_bytes": float(descriptor_bytes + view_bytes),
        "descriptors_only_pickle_bytes": float(serialized),
    }


def place_recognition_recall(
    query_descriptors: np.ndarray,
    query_xy: np.ndarray,
    db_descriptors: np.ndarray,
    db_xy: np.ndarray,
    k_values: tuple[int, ...] = (1, 5),
    distance_threshold: float = 5.0,
    query_index: np.ndarray | None = None,
    db_index: np.ndarray | None = None,
    exclude_window: int = 0,
) -> dict[str, float]:
    """Average Recall@k for visual place recognition (PRISM Table II protocol).

    For each query, the database entries are ranked by descriptor similarity
    (cosine; for L2-normalized descriptors this matches PRISM's Euclidean
    ranking). The query is a hit@k if ANY of its top-k database neighbours lies
    within ``distance_threshold`` metres of the query's true position. Recall@k
    is the fraction of queries that are a hit@k, i.e. PRISM's AR@k.

    Args:
        query_descriptors: (Q, d) query descriptors.
        query_xy: (Q, 2) query ground-truth positions.
        db_descriptors: (D, d) database descriptors.
        db_xy: (D, 2) database ground-truth positions.
        k_values: cutoffs to report (e.g. (1, 5) -> recall_at_1, recall_at_5).
        distance_threshold: metres within which a retrieved place counts as
            correct (PRISM use 5 m).
        query_index, db_index: optional 1-D frame indices. When both are given,
            database entries whose index is within ``exclude_window`` of the
            query's index are dropped BEFORE ranking. This is how the
            *within-traversal* protocol excludes near-in-time self-matches
            (PRISM exclude "frames from identical locations"); leave unset for
            the *cross-traversal* protocol, where query and database are
            different runs and no self-match exists.
        exclude_window: half-width (in index units) of the self-match exclusion.

    Returns:
        Dict ``recall_at_<k>`` for each k in ``k_values`` (NaN if no queries),
        plus ``n_queries``.
    """
    q: int = query_descriptors.shape[0]
    if q == 0 or db_descriptors.shape[0] == 0:
        return {f"recall_at_{k}": float("nan") for k in k_values} | {"n_queries": 0}

    def _l2norm(matrix: np.ndarray) -> np.ndarray:
        return matrix / np.maximum(
            np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12
        )

    queries: np.ndarray = _l2norm(query_descriptors.astype(np.float64))
    database: np.ndarray = _l2norm(db_descriptors.astype(np.float64))
    similarity: np.ndarray = queries @ database.T  # (Q, D)

    # Distance (m) from each query to every database entry, for the gate.
    deltas: np.ndarray = query_xy[:, None, :] - db_xy[None, :, :]
    distances: np.ndarray = np.linalg.norm(deltas, axis=2)  # (Q, D)

    use_exclusion: bool = (
        query_index is not None and db_index is not None and exclude_window > 0
    )
    max_k: int = min(max(k_values), database.shape[0])
    hits: dict[int, int] = {k: 0 for k in k_values}
    for i in range(q):
        scores: np.ndarray = similarity[i].copy()
        if use_exclusion:
            too_close: np.ndarray = (
                np.abs(db_index - int(query_index[i])) <= exclude_window
            )
            scores[too_close] = -np.inf
        order: np.ndarray = np.argsort(scores)[::-1][:max_k]
        correct: np.ndarray = distances[i, order] <= distance_threshold
        for k in k_values:
            if bool(correct[:k].any()):
                hits[k] += 1
    out: dict[str, float] = {f"recall_at_{k}": hits[k] / q for k in k_values}
    out["n_queries"] = q
    return out


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


def retrieval_report(
    ranked_labels: list[list[str]],
    true_labels: list[str],
    confident_flags: list[bool] | None = None,
    k: int = 3,
) -> dict[str, object]:
    """Paper-grade language-to-place retrieval evaluation.

    Args:
        ranked_labels: per query, the room labels of the returned nodes in rank
            order.
        true_labels: the correct room label per query.
        confident_flags: per query, whether the retriever committed to an answer
            (its calibrated-rejection decision). Enables coverage/precision.
        k: cutoff for Recall@k.

    Returns:
        Dict with:
        - ``overall``: recall@1, recall@k, mrr, n_queries.
        - ``per_class``: recall@1 and support (n) per ground-truth room.
        - ``confusion``: true-label -> {predicted top-1 label: count}.
        - ``rejection`` (if ``confident_flags`` given): coverage (fraction
          answered), precision@1 on answered queries, recall@1 if always
          answering, and the fraction of wrong answers correctly rejected — the
          evidence that calibrated rejection trades coverage for precision.
    """
    n: int = len(true_labels)
    if n == 0:
        return {"overall": {"recall_at_1": 0.0, f"recall_at_{k}": 0.0, "mrr": 0.0}}

    overall: dict[str, float] = retrieval_metrics(ranked_labels, true_labels, k)
    overall["n_queries"] = n

    correct_top1: list[bool] = [
        bool(ranks and ranks[0] == truth)
        for ranks, truth in zip(ranked_labels, true_labels)
    ]

    # Per ground-truth class: recall@1 and support.
    classes: list[str] = sorted(set(true_labels))
    per_class: dict[str, dict[str, float]] = {}
    for label in classes:
        idx = [i for i, t in enumerate(true_labels) if t == label]
        hits = sum(correct_top1[i] for i in idx)
        per_class[label] = {
            "recall_at_1": round(hits / len(idx), 4),
            "n": len(idx),
        }

    # Confusion: true label -> predicted top-1 label counts.
    confusion: dict[str, dict[str, int]] = {label: {} for label in classes}
    for ranks, truth in zip(ranked_labels, true_labels):
        predicted: str = ranks[0] if ranks else "?"
        confusion[truth][predicted] = confusion[truth].get(predicted, 0) + 1

    report: dict[str, object] = {
        "overall": {key: round(float(val), 4) if isinstance(val, float) else val
                    for key, val in overall.items()},
        "per_class": per_class,
        "confusion": confusion,
    }

    if confident_flags is not None:
        answered = [i for i, c in enumerate(confident_flags) if c]
        wrong = [i for i in range(n) if not correct_top1[i]]
        rejected_wrong = [i for i in wrong if not confident_flags[i]]
        report["rejection"] = {
            "coverage": round(len(answered) / n, 4),
            "precision_at_1_when_answered": round(
                sum(correct_top1[i] for i in answered) / len(answered), 4
            ) if answered else 0.0,
            "recall_at_1_if_always_answer": round(overall["recall_at_1"], 4),
            "wrong_answers_rejected_rate": round(
                len(rejected_wrong) / len(wrong), 4
            ) if wrong else 0.0,
            "n_answered": len(answered),
        }
    return report


def rejection_curve(
    correct_top1: list[bool], margins: list[float]
) -> list[dict[str, float]]:
    """Coverage vs precision as the rejection threshold sweeps.

    For each candidate top-2 posterior-margin threshold, report the fraction of
    queries answered (coverage) and the top-1 accuracy on those (precision). The
    operating curve of the calibrated-rejection rule — the figure to plot.

    Args:
        correct_top1: per query, whether the top-1 label was correct.
        margins: per query, the top-2 posterior margin the rule thresholds on.
    """
    n: int = len(correct_top1)
    if n == 0:
        return []
    thresholds: list[float] = sorted({0.0, *(round(float(m), 4) for m in margins)})
    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        answered = [i for i in range(n) if margins[i] >= threshold]
        coverage = len(answered) / n
        precision = (
            sum(correct_top1[i] for i in answered) / len(answered)
            if answered else 1.0
        )
        curve.append({
            "margin": round(float(threshold), 4),
            "coverage": round(coverage, 4),
            "precision_at_1": round(precision, 4),
        })
    return curve
