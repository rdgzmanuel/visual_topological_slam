#!/usr/bin/env bash
# Full experiment suite for the paper, with the single fixed hyperparameter
# set (valley_k / visual_outlier_k / merge_radius = 2.0 / 2.0 / 2.0 in every
# config). For each of the four COLD environments it runs:
#
#   1. The main pipeline (dual gate, pose-graph optimization ON). The mapper
#      also saves graph_*_noopt.pkl, so RMSE can be reported before AND after
#      optimization from this single run.
#   2. The gate ablation: visual-only, geometric-only, and the naive
#      absolute-threshold baseline. Outputs land in output/<env>_<mode>/.
#   3. Metrics reports (with ground truth, so RMSE / false-merge are filled):
#      - output/<env>/nlp_report.json      main run: graph metrics (RMSE opt.)
#                                          + NLP retrieval (tab:nlp)
#      - output/<env>/noopt_report.json    RMSE before optimization (tab:mapping)
#      - output/<env>_<mode>/metrics_report.json   per ablation mode (tab:ablation)
#
# Each launch exits by itself when the sequence is done (graph_builder
# exit_when_done); no manual Ctrl-C is needed.
#
# Resumable: a mapping run is SKIPPED when its final_graph.pkl already exists
# and is newer than the environment's config file (i.e. it was produced after
# the current parameters were set). The cheap evaluation steps always re-run.
#
# Usage:
#   ./run_experiments.sh                     # all four environments
#   ./run_experiments.sh saarbruecken_a ...  # only the named environments
#
# Run inside the devcontainer after `colcon build` and sourcing the overlay.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/encoder/seq_data}"

CONFIGS=(cold_freiburg_a cold_freiburg_ext cold_saarbruecken_a cold_saarbruecken_ext)
OUTDIRS=(freiburg_a freiburg_ext saarbruecken_a saarbruecken_ext)
SEQS=(cold-freiburg_part_a_seq2_night1 cold-freiburg_part_b_seq3_sunny1
      cold-saarbruecken_part_a_seq2_night2 cold-saarbruecken_part_b_seq4_sunny1)
GATE_MODES=(both visual geometric threshold)

wanted() {  # wanted <env> [filters...]; no filters = run everything
    local env="$1"; shift
    [ "$#" -eq 0 ] && return 0
    for w in "$@"; do [ "$w" = "$env" ] && return 0; done
    return 1
}

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}.yaml"
    config_path="src/vts_bringup/config/${config}"
    outdir="${OUTDIRS[$i]}"
    gt_dir="${DATA_ROOT}/${SEQS[$i]}/std_cam"
    wanted "$outdir" "$@" || continue

    for mode in "${GATE_MODES[@]}"; do
        if [ "$mode" = "both" ]; then
            result="output/${outdir}/final_graph.pkl"
        else
            result="output/${outdir}_${mode}/final_graph.pkl"
        fi
        if [ -f "$result" ] && [ "$result" -nt "$config_path" ]; then
            echo "=== ${config} | gate_mode=${mode} — up to date, skipping ==="
            continue
        fi
        echo "=== ${config} | gate_mode=${mode} ==="
        ros2 launch vts_bringup pipeline.launch.py \
            config:="${config}" mode:=building gate_mode:="${mode}"
    done

    echo "=== ${config} | NLP retrieval + optimized-map metrics ==="
    python3 -m vts_evaluation.make_queries \
        --graph "output/${outdir}/final_graph.pkl" \
        --out "output/${outdir}/queries.json"
    python3 -m vts_evaluation.evaluate_run \
        --graph "output/${outdir}/final_graph.pkl" \
        --node-gt "output/${outdir}/graph_0_node_gt.json" \
        --gt-trajectory "${gt_dir}" \
        --queries "output/${outdir}/queries.json" \
        > "output/${outdir}/nlp_report.json"

    echo "=== ${config} | RMSE before optimization ==="
    python3 -m vts_evaluation.evaluate_run \
        --graph "output/${outdir}/graph_0_noopt.pkl" \
        --node-gt "output/${outdir}/graph_0_node_gt.json" \
        --gt-trajectory "${gt_dir}" \
        --no-viz \
        > "output/${outdir}/noopt_report.json"

    for mode in visual geometric threshold; do
        echo "=== ${config} | ablation metrics: ${mode} ==="
        python3 -m vts_evaluation.evaluate_run \
            --graph "output/${outdir}_${mode}/final_graph.pkl" \
            --node-gt "output/${outdir}_${mode}/graph_0_node_gt.json" \
            --gt-trajectory "${gt_dir}" \
            --no-viz \
            > "output/${outdir}_${mode}/metrics_report.json"
    done
done

echo "All experiments done. Per-environment reports in output/*/."
