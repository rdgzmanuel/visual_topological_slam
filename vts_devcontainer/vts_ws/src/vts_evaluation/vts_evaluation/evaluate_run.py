"""Offline evaluation CLI.

Computes the metrics tables for the paper from artifacts the pipeline saved
to disk (no ROS required):

    python -m vts_evaluation.evaluate_run \
        --graph output/graphs/final_graph.pkl \
        --node-gt output/graphs/graph_0_node_gt.json \
        --gt-trajectory /path/to/cold/sequence/std_cam \
        --queries queries.json

- Graph metrics: node/edge counts, coverage, node placement RMSE,
  false-merge (perceptual aliasing) rate.
- Odometry metrics (if the player logged trajectories): ATE RMSE/median and
  final drift of the simulated odometry vs. ground truth.
- Retrieval metrics (if ``--queries`` given): Recall@1, Recall@k, MRR.
  ``queries.json`` format: [{"query": "the kitchen", "label": "KT"}, ...] —
  labels must match the room labels attached to nodes (requires a COLD
  places annotation file at mapping time).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from vts_core.metrics import graph_metrics, retrieval_metrics, trajectory_metrics
from vts_core.retrieval import PlaceRetriever, SemanticEncoder
from vts_core.topo_graph import TopoGraph


def _gt_trajectory_from_cold_dir(images_dir: str) -> np.ndarray:
    """Parse ground-truth (x, y) from COLD filenames in a directory."""
    from vts_players.cold_player_node import parse_cold_filename

    points: list[tuple[float, float]] = []
    for name in sorted(os.listdir(images_dir)):
        parsed = parse_cold_filename(name)
        if parsed is not None:
            _, pose = parsed
            points.append((pose[0], pose[1]))
    return np.array(points, dtype=np.float64).reshape(-1, 2)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, help="Path to a TopoGraph pickle")
    parser.add_argument("--node-gt", default="", help="node_gt JSON from the mapper")
    parser.add_argument(
        "--gt-trajectory",
        default="",
        help="COLD images dir to parse the GT trajectory from",
    )
    parser.add_argument(
        "--odom-trajectory",
        default="",
        help="Optional .npy (N, 2) of simulated odometry positions for ATE",
    )
    parser.add_argument("--queries", default="", help="queries.json for retrieval")
    parser.add_argument(
        "--semantic-model", default="openai/clip-vit-base-patch32"
    )
    parser.add_argument("--top-k", type=int, default=3)
    args: argparse.Namespace = parser.parse_args()

    graph: TopoGraph = TopoGraph.load(args.graph)
    report: dict[str, object] = {}

    gt_xy: np.ndarray | None = None
    if args.gt_trajectory:
        gt_xy = _gt_trajectory_from_cold_dir(args.gt_trajectory)

    node_gt: dict[int, np.ndarray] | None = None
    if args.node_gt and os.path.exists(args.node_gt):
        with open(args.node_gt) as f:
            raw: dict[str, list[float]] = json.load(f)
        node_gt = {
            int(k): np.array(v[:2], dtype=np.float64) for k, v in raw.items()
        }

    if gt_xy is not None and gt_xy.size:
        gm = graph_metrics(graph, gt_xy, node_gt_xy=node_gt)
        report["graph"] = {
            "n_nodes": gm.n_nodes,
            "n_edges": gm.n_edges,
            "coverage": round(gm.coverage, 4),
            "node_placement_rmse_m": round(gm.node_placement_rmse, 4),
            "false_merge_rate": round(gm.false_merge_rate, 4),
            "spatial_tolerance_m": round(gm.spatial_tolerance, 4),
        }
    else:
        report["graph"] = {
            "n_nodes": len(graph.nodes),
            "n_edges": len(graph.edges()),
        }

    if args.odom_trajectory and gt_xy is not None:
        odom_xy: np.ndarray = np.load(args.odom_trajectory).reshape(-1, 2)
        n: int = min(odom_xy.shape[0], gt_xy.shape[0])
        tm = trajectory_metrics(odom_xy[:n], gt_xy[:n])
        report["odometry"] = {
            "ate_rmse_m": round(tm.ate_rmse, 4),
            "ate_median_m": round(tm.ate_median, 4),
            "final_drift_m": round(tm.final_drift, 4),
        }

    if args.queries:
        with open(args.queries) as f:
            queries: list[dict[str, str]] = json.load(f)
        retriever: PlaceRetriever = PlaceRetriever(
            SemanticEncoder(args.semantic_model), graph
        )
        ranked_labels: list[list[str]] = []
        true_labels: list[str] = []
        for entry in queries:
            ranked, _ = retriever.query(entry["query"], top_k=args.top_k)
            ranked_labels.append(
                [node.room_label or "?" for node, _ in ranked]
            )
            true_labels.append(entry["label"])
        report["retrieval"] = retrieval_metrics(
            ranked_labels, true_labels, k=args.top_k
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
