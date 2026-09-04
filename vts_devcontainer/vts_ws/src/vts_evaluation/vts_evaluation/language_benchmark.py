"""Evaluate one language query set across several topological maps.

Unlike the older retrieval evaluation, this benchmark also asks for room
types that are absent from a map.  It therefore measures both retrieval when
a destination exists and rejection when it does not.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vts_core.retrieval import PlaceRetriever, SemanticEncoder
from vts_core.topo_graph import TopoGraph


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _round(value: float) -> float:
    return round(float(value), 4)


def _present_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    reciprocal_ranks: list[float] = []
    for row in rows:
        labels: list[str] = row["ranked_labels"]
        try:
            reciprocal_ranks.append(1.0 / (labels.index(row["label"]) + 1))
        except ValueError:
            reciprocal_ranks.append(0.0)

    answered: list[dict[str, Any]] = [row for row in rows if row["confident"]]
    return {
        "n": len(rows),
        "recall_at_1": _round(_mean([row["top1_correct"] for row in rows])),
        "recall_at_3": _round(_mean([row["top3_correct"] for row in rows])),
        "mrr": _round(_mean(reciprocal_ranks)),
        "answer_coverage": _round(len(answered) / len(rows)) if rows else 0.0,
        "precision_at_1_when_answered": _round(
            _mean([row["top1_correct"] for row in answered])
        ),
        "successful_answer_rate": _round(
            _mean([row["confident"] and row["top1_correct"] for row in rows])
        ),
    }


def _absent_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        "correct_rejection_rate": _round(
            _mean([not row["confident"] for row in rows])
        ),
        "false_answer_rate": _round(_mean([row["confident"] for row in rows])),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    present: list[dict[str, Any]] = [row for row in rows if row["present"]]
    absent: list[dict[str, Any]] = [row for row in rows if not row["present"]]
    safe_outcomes: list[bool] = [
        bool(row["confident"] and row["top1_correct"])
        if row["present"]
        else not bool(row["confident"])
        for row in rows
    ]
    return {
        "n_map_query_pairs": len(rows),
        "present": _present_metrics(present),
        "absent": _absent_metrics(absent),
        "safe_decision_accuracy": _round(_mean(safe_outcomes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--maps-root", required=True)
    parser.add_argument("--graph-name", default="graph_0_noopt.pkl")
    parser.add_argument(
        "--semantic-model", default="openai/clip-vit-base-patch32"
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.benchmark, encoding="utf-8") as stream:
        benchmark: dict[str, Any] = json.load(stream)

    queries: list[dict[str, str]] = benchmark["queries"]
    if len(queries) != 100:
        raise ValueError(f"Expected 100 queries, found {len(queries)}")

    encoder = SemanticEncoder(args.semantic_model)
    all_rows: list[dict[str, Any]] = []
    environment_reports: dict[str, Any] = {}

    for environment, labels in benchmark["environments"].items():
        graph_path = Path(args.maps_root) / environment / args.graph_name
        graph = TopoGraph.load(str(graph_path))
        retriever = PlaceRetriever(encoder, graph)
        expected_labels = set(labels)
        rows: list[dict[str, Any]] = []

        for query in queries:
            ranked, confident = retriever.query(query["query"], top_k=args.top_k)
            ranked_labels = [node.room_label or "?" for node, _, _ in ranked]
            posterior = [probability for _, probability, _ in ranked]
            row: dict[str, Any] = {
                "environment": environment,
                **query,
                "present": query["label"] in expected_labels,
                "ranked_labels": ranked_labels,
                "top1_correct": bool(
                    ranked_labels and ranked_labels[0] == query["label"]
                ),
                "top3_correct": query["label"] in ranked_labels[:3],
                "confident": bool(confident),
                "posterior_margin": _round(
                    posterior[0] - posterior[1]
                    if len(posterior) >= 2
                    else 1.0
                ),
            }
            rows.append(row)
            all_rows.append(row)

        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["present"]:
                by_category[row["category"]].append(row)
        environment_reports[environment] = {
            **_summarize(rows),
            "present_by_category": {
                category: _present_metrics(category_rows)
                for category, category_rows in sorted(by_category.items())
            },
        }

    by_category_all: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        if row["present"]:
            by_category_all[row["category"]].append(row)

    report: dict[str, Any] = {
        "benchmark": {
            "name": benchmark["name"],
            "version": benchmark["version"],
            "n_queries": len(queries),
            "n_environments": len(benchmark["environments"]),
        },
        "semantic_model": args.semantic_model,
        "overall": {
            **_summarize(all_rows),
            "present_by_category": {
                category: _present_metrics(category_rows)
                for category, category_rows in sorted(by_category_all.items())
            },
        },
        "environments": environment_reports,
        "results": all_rows,
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"overall": report["overall"], "environments": environment_reports}, indent=2))


if __name__ == "__main__":
    main()
