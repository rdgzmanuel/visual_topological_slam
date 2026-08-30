"""Offline evaluation CLI.

Computes the metrics tables for the paper from artifacts the pipeline saved
to disk (no ROS required):

    python -m vts_evaluation.evaluate_run \
        --graph output/graphs/final_graph.pkl \
        --node-gt output/graphs/graph_0_node_gt.json \
        --gt-trajectory /path/to/ground-truth \
        --queries queries.json

- Graph metrics: node/edge counts, coverage and node placement RMSE.
- Loop closures: accepted/rejected candidates, TP/FP/FN, precision, recall,
  F1, false shortcuts and duplicate nodes. Sequential edges are kept out of
  the closure denominator.
- Topology diagnostics: degree statistics and median path distortion.
- Descriptor diagnostics: intra- vs inter-room descriptor similarity and the
  nearest-neighbour same-room rate (needs room labels). High separation with
  a low NN-rate and a high max degree is the signature of map corruption with
  a healthy encoder — distinguishes a mapping bug from a weak encoder.
- Odometry metrics (if a synchronized trajectory is supplied): ATE
  RMSE/median and final drift of recorded odometry vs. ground truth.
- Retrieval metrics (if ``--queries`` given): Recall@1, Recall@k, MRR.
  ``queries.json`` format: [{"query": "the kitchen", "label": "KT"}, ...] —
  labels must match the room labels attached to nodes (requires a COLD
  places annotation file at mapping time).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import redirect_stdout

import numpy as np

from vts_core.metrics import (
    descriptor_separation,
    directional_model_statistics,
    graph_metrics,
    rejection_curve,
    retrieval_report,
    trajectory_metrics,
)
from vts_core.topo_graph import TopoGraph
from vts_evaluation.ground_truth import load_ground_truth_xy


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, help="Path to a TopoGraph pickle")
    parser.add_argument("--node-gt", default="", help="node_gt JSON from the mapper")
    parser.add_argument(
        "--gt-trajectory",
        default="",
        help="COLD std_cam directory or CID-SIMS groundtruth.txt",
    )
    parser.add_argument(
        "--odom-trajectory",
        default="",
        help="Optional aligned .npy (N, 2) of recorded odometry for ATE",
    )
    parser.add_argument("--queries", default="", help="queries.json for retrieval")
    parser.add_argument(
        "--report-rejection",
        action="store_true",
        help=(
            "Report the exploratory fixed-margin rejection heuristic. It is "
            "omitted by default because it has not been calibrated on held-out data."
        ),
    )
    parser.add_argument(
        "--semantic-model", default="openai/clip-vit-base-patch32"
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--loop-tolerance",
        type=float,
        default=2.0,
        help="GT distance (m) defining a true loop closure (default: 2.0)",
    )
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
        "--floorplan-viz-path",
        default="",
        help=(
            "Output path for the calibrated floorplan overlay. Defaults to "
            "<graph_dir>/images/<stem>_on_floorplan.png; use a .pdf path for "
            "a publication figure."
        ),
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Disable the map images"
    )
    args: argparse.Namespace = parser.parse_args()

    graph: TopoGraph = TopoGraph.load(args.graph)
    report: dict[str, object] = {}

    gt_xy: np.ndarray | None = None
    if args.gt_trajectory:
        gt_xy = load_ground_truth_xy(args.gt_trajectory)

    node_gt: dict[int, np.ndarray] | None = None
    if args.node_gt and os.path.exists(args.node_gt):
        with open(args.node_gt) as f:
            raw: dict[str, list[float]] = json.load(f)
        node_gt = {
            int(k): np.array(v[:2], dtype=np.float64) for k, v in raw.items()
        }

    if gt_xy is not None and gt_xy.size:
        gm = graph_metrics(
            graph, gt_xy, node_gt_xy=node_gt,
            loop_tolerance=args.loop_tolerance,
        )
        report["graph"] = {
            "n_nodes": gm.n_nodes,
            "n_edges": gm.n_edges,
            "coverage": round(gm.coverage, 4),
            "coverage_1m": round(gm.coverage_1m, 4),
            "coverage_2m": round(gm.coverage_2m, 4),
            "node_placement_rmse_m": round(gm.node_placement_rmse, 4),
            "spatial_tolerance_m": round(gm.spatial_tolerance, 4),
            "max_degree": gm.max_degree,
            "mean_degree": round(gm.mean_degree, 4),
            "n_components": gm.n_components,
            "n_sequential_edges": gm.n_sequential_edges,
            "n_loop_edges": gm.n_loop_edges,
            "loop_candidates_accepted": gm.accepted_candidates,
            "loop_candidates_rejected": gm.rejected_candidates,
            "loop_tp": gm.loop_true_positives,
            "loop_fp": gm.loop_false_positives,
            "loop_fn": gm.loop_false_negatives,
            "loop_precision": round(gm.loop_precision, 4),
            "loop_recall": round(gm.loop_recall, 4),
            "loop_f1": round(gm.loop_f1, 4),
            "semantic_loop_evaluable": gm.semantic_loop_evaluable,
            "semantic_loop_correct": gm.semantic_loop_correct,
            "semantic_loop_shortcuts": gm.semantic_loop_shortcuts,
            "semantic_loop_precision": round(gm.semantic_loop_precision, 4),
            "duplicate_nodes": gm.duplicate_nodes,
            "false_shortcuts": gm.false_shortcuts,
            "median_path_distortion": round(gm.median_path_distortion, 4),
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
            "n_sequential_edges": gm.n_sequential_edges,
            "n_loop_edges": gm.n_loop_edges,
        }

    separation: dict[str, float] = descriptor_separation(graph)
    if separation:
        report["descriptors"] = separation
    directional_stats: dict[str, float | int] = (
        directional_model_statistics(graph)
    )
    if directional_stats:
        report["directional_visual_model"] = directional_stats

    if not args.no_viz:
        from vts_evaluation.map_viz import render_map, render_on_floorplan

        images_dir: str = os.path.join(
            os.path.dirname(os.path.abspath(args.graph)), "images"
        )
        os.makedirs(images_dir, exist_ok=True)
        stem: str = os.path.splitext(os.path.basename(args.graph))[0]
        viz_path: str = args.viz_path or os.path.join(images_dir, f"{stem}_map.png")
        # stdout is the machine-readable JSON contract of this CLI. Plotting
        # status messages belong on stderr so shell redirection cannot corrupt
        # the report.
        with redirect_stdout(sys.stderr):
            render_map(graph, viz_path, gt_xy=gt_xy, node_gt=node_gt)
        report["map_image"] = viz_path

        if args.floorplan and node_gt is not None:
            overlay_path: str = args.floorplan_viz_path or os.path.join(
                images_dir, f"{stem}_on_floorplan.png",
            )
            overlay_dir: str = os.path.dirname(os.path.abspath(overlay_path))
            os.makedirs(overlay_dir, exist_ok=True)
            with redirect_stdout(sys.stderr):
                rendered = render_on_floorplan(
                    graph, args.floorplan, overlay_path,
                    node_gt=node_gt, gt_xy=gt_xy,
                )
            if rendered:
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
        indexing_start = time.perf_counter()
        retriever: PlaceRetriever = PlaceRetriever(
            SemanticEncoder(args.semantic_model), graph
        )
        indexing_time_s = time.perf_counter() - indexing_start
        ranked_labels: list[list[str]] = []
        true_labels: list[str] = []
        confident_flags: list[bool] = []
        margins: list[float] = []
        query_times_s: list[float] = []
        for entry in queries:
            query_start = time.perf_counter()
            ranked, confident = retriever.query(entry["query"], top_k=args.top_k)
            query_times_s.append(time.perf_counter() - query_start)
            ranked_labels.append(
                [node.room_label or "?" for node, _, _ in ranked]
            )
            true_labels.append(entry["label"])
            confident_flags.append(confident)
            margins.append(
                float(ranked[0][1] - ranked[1][1]) if len(ranked) >= 2 else 1.0
            )
        report["retrieval"] = retrieval_report(
            ranked_labels,
            true_labels,
            confident_flags if args.report_rejection else None,
            k=args.top_k,
        )
        if args.report_rejection:
            correct_top1 = [
                bool(r and r[0] == t)
                for r, t in zip(ranked_labels, true_labels, strict=True)
            ]
            report["retrieval"]["rejection_curve"] = rejection_curve(
                correct_top1, margins
            )
        report["retrieval"]["runtime"] = {
            "model_load_and_map_index_s": round(indexing_time_s, 4),
            "mean_query_ms": round(
                1000.0 * float(np.mean(query_times_s)), 4
            ) if query_times_s else 0.0,
        }
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
            with redirect_stdout(sys.stderr):
                confusion_rendered = render_confusion(
                    report["retrieval"]["confusion"], conf_path
                )
            if confusion_rendered:
                report["retrieval"]["confusion_image"] = conf_path
            if args.report_rejection:
                curve_path = os.path.join(images_dir, "retrieval_rejection_curve.png")
                with redirect_stdout(sys.stderr):
                    curve_rendered = render_rejection_curve(
                        report["retrieval"]["rejection_curve"], curve_path
                    )
                if curve_rendered:
                    report["retrieval"]["rejection_curve_image"] = curve_path

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
