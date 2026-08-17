"""Aggregate revised experiment reports into paper-ready JSON and CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import os

ENVIRONMENTS = (
    "freiburg_a",
    "freiburg_ext",
    "saarbruecken_a",
    "saarbruecken_ext",
    "cid_sims_apartment1_1",
    "cid_sims_apartment2_1",
    "cid_sims_apartment3_1",
)
GATE_MODES = ("both", "visual", "geometric", "threshold")


def _load(path: str) -> dict[str, object]:
    with open(path) as handle:
        return json.load(handle)


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="output/revised")
    args = parser.parse_args()

    summary: dict[str, object] = {"environments": {}}
    mapping_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    retrieval_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []

    for environment in ENVIRONMENTS:
        main_dir = os.path.join(args.root, environment)
        final_path = os.path.join(main_dir, "metrics_report.json")
        gtsam_path = os.path.join(main_dir, "gtsam_metrics_report.json")
        performance_path = os.path.join(
            main_dir, "graph_0_performance.json"
        )
        if not all(
            os.path.exists(path)
            for path in (final_path, performance_path)
        ):
            continue
        final = _load(final_path)
        gtsam = _load(gtsam_path) if os.path.exists(gtsam_path) else {}
        performance = _load(performance_path)
        gates: dict[str, object] = {"both": final["graph"]}
        for mode in GATE_MODES[1:]:
            report_path = os.path.join(
                args.root,
                f"{environment}_{mode}",
                "metrics_report.json",
            )
            if (
                not os.path.exists(report_path)
                or os.path.getmtime(report_path) < os.path.getmtime(final_path)
            ):
                continue
            report = _load(report_path)
            gates[mode] = report["graph"]

        environment_summary = {
            "final_graph": final["graph"],
            "gtsam_ablation": gtsam.get("graph", {}),
            "gate_ablations": gates,
            "retrieval": final.get("retrieval", {}),
            "performance": performance,
        }
        summary["environments"][environment] = environment_summary

        final_graph = final["graph"]
        gtsam_graph = gtsam.get("graph", {})
        mapping_rows.append(
            {
                "environment": environment,
                "n_nodes": final_graph.get("n_nodes"),
                "n_edges": final_graph.get("n_edges"),
                "rmse_final_m": final_graph.get("node_placement_rmse_m"),
                "rmse_gtsam_m": gtsam_graph.get("node_placement_rmse_m"),
                "coverage_1m": final_graph.get("coverage_1m"),
                "coverage_2m": final_graph.get("coverage_2m"),
                "duplicate_nodes": final_graph.get("duplicate_nodes"),
                "false_shortcuts": final_graph.get("false_shortcuts"),
                "semantic_loop_precision": final_graph.get(
                    "semantic_loop_precision"
                ),
                "semantic_loop_shortcuts": final_graph.get(
                    "semantic_loop_shortcuts"
                ),
                "median_path_distortion": final_graph.get(
                    "median_path_distortion"
                ),
            }
        )

        for mode, graph_report in gates.items():
            ablation_rows.append(
                {"environment": environment, "gate_mode": mode, **graph_report}
            )

        retrieval = final.get("retrieval", {})
        overall = retrieval.get("overall", {})
        rejection = retrieval.get("rejection", {})
        retrieval_rows.append(
            {
                "environment": environment,
                **overall,
                "answer_coverage": rejection.get("coverage"),
                "precision_at_1_when_answered": rejection.get(
                    "precision_at_1_when_answered"
                ),
            }
        )
        runtime_rows.append({"environment": environment, **performance})

    os.makedirs(args.root, exist_ok=True)
    summary_path = os.path.join(args.root, "summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    _write_csv(os.path.join(args.root, "mapping.csv"), mapping_rows)
    _write_csv(os.path.join(args.root, "gate_ablation.csv"), ablation_rows)
    _write_csv(os.path.join(args.root, "retrieval.csv"), retrieval_rows)
    _write_csv(os.path.join(args.root, "runtime.csv"), runtime_rows)
    print(f"Aggregated results written below {args.root}")


if __name__ == "__main__":
    main()
