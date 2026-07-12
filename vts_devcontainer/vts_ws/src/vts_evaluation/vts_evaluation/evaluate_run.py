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
- Structural diagnostics: max/mean node degree (a high max degree flags a
  perceptual-aliasing *hub*), connected components, loop-closure count
  (circuit rank), and edge-length distribution (a huge max edge length flags
  a cross-map false closure). No ground truth needed.
- Descriptor diagnostics: intra- vs inter-room descriptor similarity and the
  nearest-neighbour same-room rate (needs room labels). High separation with
  a low NN-rate and a high max degree is the signature of map corruption with
  a healthy encoder — distinguishes a mapping bug from a weak encoder.
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
import re

import numpy as np

# COLD ground-truth poses are encoded in the image filenames. Parsed here with
# a local regex so the offline evaluator needs neither ROS nor OpenCV (the
# player node that also parses these names imports both).
_COLD_FILENAME: re.Pattern[str] = re.compile(
    r"_x(?P<x>-?\d+\.?\d*)_y(?P<y>-?\d+\.?\d*)_a(?P<a>-?\d+\.?\d*)"
)

from vts_core.metrics import (
    descriptor_separation,
    graph_metrics,
    rejection_curve,
    retrieval_report,
    trajectory_metrics,
)
from vts_core.topo_graph import TopoGraph


def _gt_trajectory_from_cold_dir(images_dir: str) -> np.ndarray:
    """Parse ground-truth (x, y) from COLD filenames in a directory."""
    points: list[tuple[float, float]] = []
    for name in sorted(os.listdir(images_dir)):
        match: re.Match[str] | None = _COLD_FILENAME.search(name)
        if match is not None:
            points.append((float(match.group("x")), float(match.group("y"))))
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
    parser.add_argument(
        "--viz-path",
        default="",
        help="Where to save the map PNG (default: <graph_dir>/images/<stem>_map.png)",
    )
    parser.add_argument(
        "--floorplan",
        default="",
        help="Floorplan image to overlay the map on (needs a <floorplan>.calib.json)",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Disable the map images"
    )
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
            "coverage_1m": round(gm.coverage_1m, 4),
            "coverage_2m": round(gm.coverage_2m, 4),
            "node_placement_rmse_m": round(gm.node_placement_rmse, 4),
            "false_merge_rate": round(gm.false_merge_rate, 4),
            "spatial_tolerance_m": round(gm.spatial_tolerance, 4),
            "max_degree": gm.max_degree,
            "mean_degree": round(gm.mean_degree, 4),
            "n_components": gm.n_components,
            "n_loop_closures": gm.n_loop_closures,
            "median_edge_length_m": round(gm.median_edge_length, 4),
            "max_edge_length_m": round(gm.max_edge_length, 4),
        }
    else:
        gm = graph_metrics(graph, graph.positions())
        report["graph"] = {
            "n_nodes": gm.n_nodes,
            "n_edges": gm.n_edges,
            "max_degree": gm.max_degree,
            "mean_degree": round(gm.mean_degree, 4),
            "n_components": gm.n_components,
            "n_loop_closures": gm.n_loop_closures,
        }

    separation: dict[str, float] = descriptor_separation(graph)
    if separation:
        report["descriptors"] = separation

    if not args.no_viz:
        from vts_evaluation.map_viz import render_map, render_on_floorplan

        images_dir: str = os.path.join(
            os.path.dirname(os.path.abspath(args.graph)), "images"
        )
        os.makedirs(images_dir, exist_ok=True)
        stem: str = os.path.splitext(os.path.basename(args.graph))[0]
        graph_report = report["graph"]
        title: str = (
            f"{os.path.basename(args.graph)} | "
            f"nodes={graph_report['n_nodes']} edges={graph_report['n_edges']}"
            f" | Ncomp={graph_report['n_components']}"
        )
        if "node_placement_rmse_m" in graph_report:
            title += (
                f" | placement RMSE={graph_report['node_placement_rmse_m']}m"
                f" | false-merge={graph_report['false_merge_rate']}"
                f" | max-deg={graph_report['max_degree']}"
            )

        viz_path: str = args.viz_path or os.path.join(images_dir, f"{stem}_map.png")
        render_map(graph, viz_path, gt_xy=gt_xy, node_gt=node_gt, title=title)
        report["map_image"] = viz_path

        if args.floorplan and node_gt is not None:
            overlay_path: str = os.path.join(
                images_dir, f"{stem}_on_floorplan.png"
            )
            if render_on_floorplan(
                graph, args.floorplan, overlay_path,
                node_gt=node_gt, gt_xy=gt_xy, title=title,
            ):
                report["floorplan_image"] = overlay_path

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
        # Lazy import: graph/descriptor diagnostics above need only NumPy, so
        # the heavy CLIP stack (torch/transformers) is required only here.
        from vts_core.retrieval import PlaceRetriever, SemanticEncoder

        with open(args.queries) as f:
            queries: list[dict[str, str]] = json.load(f)
        retriever: PlaceRetriever = PlaceRetriever(
            SemanticEncoder(args.semantic_model), graph
        )
        ranked_labels: list[list[str]] = []
        true_labels: list[str] = []
        confident_flags: list[bool] = []
        margins: list[float] = []
        for entry in queries:
            ranked, confident = retriever.query(entry["query"], top_k=args.top_k)
            ranked_labels.append(
                [node.room_label or "?" for node, _, _ in ranked]
            )
            true_labels.append(entry["label"])
            confident_flags.append(confident)
            margins.append(
                float(ranked[0][1] - ranked[1][1]) if len(ranked) >= 2 else 1.0
            )
        report["retrieval"] = retrieval_report(
            ranked_labels, true_labels, confident_flags, k=args.top_k
        )
        correct_top1 = [
            bool(r and r[0] == t) for r, t in zip(ranked_labels, true_labels)
        ]
        report["retrieval"]["rejection_curve"] = rejection_curve(
            correct_top1, margins
        )
        if not args.no_viz:
            from vts_evaluation.map_viz import (
                render_confusion,
                render_rejection_curve,
            )

            images_dir = os.path.join(
                os.path.dirname(os.path.abspath(args.graph)), "images"
            )
            os.makedirs(images_dir, exist_ok=True)
            conf_path = os.path.join(images_dir, "retrieval_confusion.png")
            if render_confusion(report["retrieval"]["confusion"], conf_path):
                report["retrieval"]["confusion_image"] = conf_path
            curve_path = os.path.join(images_dir, "retrieval_rejection_curve.png")
            if render_rejection_curve(
                report["retrieval"]["rejection_curve"], curve_path
            ):
                report["retrieval"]["rejection_curve_image"] = curve_path

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
