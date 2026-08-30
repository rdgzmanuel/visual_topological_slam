"""Re-score completed maps under several ground-truth distance tolerances.

The sweep is evaluation-only: it loads existing graph artifacts and changes
neither mapping decisions nor model parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

from vts_core.metrics import GraphMetrics, graph_metrics
from vts_core.topo_graph import TopoGraph
from vts_evaluation.ground_truth import load_ground_truth_xy


@dataclass(frozen=True)
class EvaluationCase:
    """Paths required to re-score one completed mapping run."""

    name: str
    graph_path: str
    node_gt_path: str
    trajectory_gt_path: str


def validate_tolerances(values: list[float]) -> list[float]:
    """Return positive, finite tolerances without duplicates."""
    tolerances: list[float] = []
    for value in values:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Distance tolerances must be positive and finite")
        if value not in tolerances:
            tolerances.append(value)
    if not tolerances:
        raise ValueError("At least one distance tolerance is required")
    return tolerances


def _load_node_ground_truth(path: str) -> dict[int, np.ndarray]:
    with open(path) as handle:
        raw: dict[str, list[float]] = json.load(handle)
    return {
        int(node_id): np.asarray(position[:2], dtype=np.float64)
        for node_id, position in raw.items()
    }


def _result_row(
    case_name: str,
    tolerance: float,
    metrics: GraphMetrics,
) -> dict[str, str | int | float]:
    return {
        "environment": case_name,
        "tolerance_m": tolerance,
        "n_nodes": metrics.n_nodes,
        "n_loop_edges": metrics.n_loop_edges,
        "coverage_at_tolerance": round(metrics.coverage, 4),
        "loop_opportunities": (
            metrics.loop_true_positives + metrics.loop_false_negatives
        ),
        "loop_tp": metrics.loop_true_positives,
        "loop_fp": metrics.loop_false_positives,
        "loop_fn": metrics.loop_false_negatives,
        "loop_precision": round(metrics.loop_precision, 4),
        "loop_recall": round(metrics.loop_recall, 4),
        "loop_f1": round(metrics.loop_f1, 4),
        "duplicate_nodes": metrics.duplicate_nodes,
        "false_shortcuts": metrics.false_shortcuts,
    }


def evaluate_case(
    case: EvaluationCase,
    tolerances: list[float],
) -> list[dict[str, str | int | float]]:
    """Evaluate one unchanged graph at each requested metric tolerance."""
    for path in (
        case.graph_path,
        case.node_gt_path,
        case.trajectory_gt_path,
    ):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input for {case.name}: {path}")

    graph: TopoGraph = TopoGraph.load(case.graph_path)
    node_gt: dict[int, np.ndarray] = _load_node_ground_truth(case.node_gt_path)
    trajectory_gt: np.ndarray = load_ground_truth_xy(case.trajectory_gt_path)
    if trajectory_gt.size == 0:
        raise ValueError(f"Empty ground-truth trajectory for {case.name}")

    rows: list[dict[str, str | int | float]] = []
    for tolerance in tolerances:
        metrics: GraphMetrics = graph_metrics(
            graph,
            trajectory_gt,
            node_gt_xy=node_gt,
            spatial_tolerance=tolerance,
            loop_tolerance=tolerance,
        )
        rows.append(_result_row(case.name, tolerance, metrics))
    return rows


def _write_csv(
    path: str,
    rows: list[dict[str, str | int | float]],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(
    path: str,
    tolerances: list[float],
    cases: list[EvaluationCase],
    rows: list[dict[str, str | int | float]],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    results_by_case: dict[str, list[dict[str, str | int | float]]] = {
        case.name: [] for case in cases
    }
    for row in rows:
        results_by_case[str(row["environment"])].append(row)

    report: dict[str, object] = {
        "protocol": {
            "tolerances_m": tolerances,
            "mapping_recomputed": False,
            "coverage_and_loop_scoring_use_same_tolerance": True,
        },
        "cases": {
            case.name: {
                "graph": case.graph_path,
                "node_ground_truth": case.node_gt_path,
                "trajectory_ground_truth": case.trajectory_gt_path,
                "results": results_by_case[case.name],
            }
            for case in cases
        },
    }
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tolerances",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 3.0],
        help="Ground-truth distance tolerances in metres",
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs=4,
        required=True,
        metavar=("NAME", "GRAPH", "NODE_GT", "TRAJECTORY_GT"),
        help="Evaluation case; repeat this option for multiple maps",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    args: argparse.Namespace = parser.parse_args()

    try:
        tolerances: list[float] = validate_tolerances(args.tolerances)
    except ValueError as error:
        parser.error(str(error))

    cases: list[EvaluationCase] = [
        EvaluationCase(*case_values) for case_values in args.case
    ]
    names: list[str] = [case.name for case in cases]
    if len(names) != len(set(names)):
        parser.error("Case names must be unique")

    rows: list[dict[str, str | int | float]] = []
    for case in cases:
        rows.extend(evaluate_case(case, tolerances))

    if args.output_json:
        _write_json(args.output_json, tolerances, cases, rows)
    if args.output_csv:
        _write_csv(args.output_csv, rows)
    if not args.output_json and not args.output_csv:
        json.dump(rows, fp=sys.stdout, indent=2)
        print()

    destinations: list[str] = [
        path for path in (args.output_json, args.output_csv) if path
    ]
    if destinations:
        print(f"Tolerance sweep written to {', '.join(destinations)}")


if __name__ == "__main__":
    main()
