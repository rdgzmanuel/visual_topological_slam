#!/usr/bin/env bash
# Revised experiment suite with the single fixed hyperparameter
# set (valley_k / visual_outlier_k = 2.0 / 2.0 in every
# config). For each of the four COLD environments it runs:
#
#   1. The final dual-gate pipeline without pose-graph optimization. Preserved
#      GTSAM artifacts provide the optimization ablation reported in the paper.
#   2. Optional gate ablations: visual-only, geometric-only, and the naive
#      absolute-threshold baseline. Set RUN_ABLATIONS=1 after inspecting the
#      main result.
#   3. Metrics reports (loop closures are scored separately from odometry):
#      - output/revised/<env>/metrics_report.json
#      - output/revised/<env>/noopt_report.json
#      - output/revised/<env>_<mode>/metrics_report.json
#   4. Publication-size figures for the main run:
#      - figures/topological_map.pdf
#      - figures/lambda2_valleys.pdf
#      - figures/recorded_odometry.pdf
#
# Each launch exits by itself when the sequence is done (graph_builder
# exit_when_done); no manual Ctrl-C is needed.
#
# Resumable: a mapping run is skipped only when its final graph is newer than
# the config and all mapping source files. Set FORCE=1 to rebuild every graph.
# Set SKIP_EXISTING=1 after a runner-only/shutdown fix to preserve every
# already completed graph regardless of source timestamps.
#
# Usage:
#   ./run_experiments.sh                     # all four environments
#   ./run_experiments.sh saarbruecken_a ...  # only the named environments
#   SKIP_EXISTING=1 ./run_experiments.sh     # resume without rebuilding graphs
#
# Run inside the devcontainer after `colcon build` and sourcing the overlay.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/encoder/seq_data}"
OUTPUT_ROOT="output/revised"
VARIANT_SUFFIX="${VARIANT_SUFFIX:-}"
FEATURE_BACKEND="${FEATURE_BACKEND:-dino_cls}"
VISUAL_MODEL="${VISUAL_MODEL:-}"
DINO_MODEL="${DINO_MODEL:-}"
DINO_LAYER="${DINO_LAYER:-}"

CONFIGS=(cold_freiburg_a cold_freiburg_ext cold_saarbruecken_a cold_saarbruecken_ext)
OUTDIRS=(freiburg_a freiburg_ext saarbruecken_a saarbruecken_ext)
SEQS=(cold-freiburg_part_a_seq2_night1 cold-freiburg_part_b_seq3_sunny1
      cold-saarbruecken_part_a_seq2_night2 cold-saarbruecken_part_b_seq4_sunny1)
GATE_MODES=(both)
if [ "${RUN_ABLATIONS:-0}" = "1" ]; then
    GATE_MODES+=(visual geometric threshold)
fi

wanted() {  # wanted <env> [filters...]; no filters = run everything
    local env="$1"; shift
    [ "$#" -eq 0 ] && return 0
    for w in "$@"; do [ "$w" = "$env" ] && return 0; done
    return 1
}

mapping_is_current() {  # mapping_is_current <result> <config>
    local result="$1"
    local config="$2"
    [ "${FORCE:-0}" != "1" ] || return 1
    [ -f "$result" ] || return 1
    [ "${SKIP_EXISTING:-0}" != "1" ] || return 0
    [ "$result" -nt "$config" ] || return 1
    while IFS= read -r source; do
        [ "$result" -nt "$source" ] || return 1
    done < <(find src/vts_core src/vts_mapping src/vts_players src/vts_bringup \
        -type f \( -name '*.py' -o -name '*.yaml' \) -print)
    return 0
}

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}.yaml"
    config_path="src/vts_bringup/config/${config}"
    outdir="${OUTDIRS[$i]}"
    base_outdir="${outdir}"
    outdir="${outdir}${VARIANT_SUFFIX}"
    gt_dir="${DATA_ROOT}/${SEQS[$i]}/std_cam"
    labels_file="${DATA_ROOT}/${SEQS[$i]}/localization/places.lst"
    wanted "$base_outdir" "$@" || continue

    main_dir="${OUTPUT_ROOT}/${outdir}"
    performance="${main_dir}/graph_0_performance.json"
    if [ -z "${VARIANT_SUFFIX}" ] && \
       [ -f "${main_dir}/metrics_report.json" ] && \
       [ ! -f "${main_dir}/visual_first_metrics_report.json" ]; then
        cp -p "${main_dir}/metrics_report.json" \
            "${main_dir}/visual_first_metrics_report.json"
    fi
    if [ -z "${VARIANT_SUFFIX}" ] && \
       [ -f "${main_dir}/metrics_report.json" ] && \
       [ ! -f "${main_dir}/always_visual_metrics_report.json" ]; then
        cp -p "${main_dir}/metrics_report.json" \
            "${main_dir}/always_visual_metrics_report.json"
    fi
    if [ -z "${VARIANT_SUFFIX}" ] && \
       [ -f "${main_dir}/metrics_report.json" ] && \
       [ ! -f "${main_dir}/adaptive_cosine_metrics_report.json" ]; then
        cp -p "${main_dir}/metrics_report.json" \
            "${main_dir}/adaptive_cosine_metrics_report.json"
    fi
    if [ -z "${VARIANT_SUFFIX}" ] && [ -f "${performance}" ] && \
       grep -q '"optimizer_backend": "gtsam"' "${performance}"; then
        [ -f "${main_dir}/gtsam_metrics_report.json" ] || \
            cp -p "${main_dir}/metrics_report.json" \
                "${main_dir}/gtsam_metrics_report.json"
        [ -f "${main_dir}/gtsam_graph.pkl" ] || \
            cp -p "${main_dir}/final_graph.pkl" "${main_dir}/gtsam_graph.pkl"
        [ -f "${main_dir}/gtsam_performance.json" ] || \
            cp -p "${performance}" "${main_dir}/gtsam_performance.json"
    fi

    for mode in "${GATE_MODES[@]}"; do
        if [ "$mode" = "both" ]; then
            result="${OUTPUT_ROOT}/${outdir}/final_graph.pkl"
        else
            result="${OUTPUT_ROOT}/${base_outdir}_${mode}${VARIANT_SUFFIX}/final_graph.pkl"
        fi
        if mapping_is_current "$result" "$config_path"; then
            echo "=== ${config} | gate_mode=${mode} — up to date, skipping ==="
            continue
        fi
        echo "=== ${config} | gate_mode=${mode} ==="
        launch_args=(
            config:="${config}"
            mode:=building
            gate_mode:="${mode}"
            feature_backend:="${FEATURE_BACKEND}"
        )
        [ -z "${VARIANT_SUFFIX}" ] || \
            launch_args+=(variant_suffix:="${VARIANT_SUFFIX}")
        [ -z "${DINO_MODEL}" ] || launch_args+=(dino_model:="${DINO_MODEL}")
        [ -z "${DINO_LAYER}" ] || launch_args+=(dino_layer:="${DINO_LAYER}")
        [ -z "${VISUAL_MODEL}" ] || \
            launch_args+=(visual_model:="${VISUAL_MODEL}")
        ros2 launch vts_bringup pipeline.launch.py "${launch_args[@]}"
    done

    echo "=== ${config} | final no-GTSAM map + text retrieval metrics ==="
    figures_dir="${OUTPUT_ROOT}/${outdir}/figures"
    main_graph="${OUTPUT_ROOT}/${outdir}/graph_0_noopt.pkl"
    mkdir -p "${figures_dir}"
    python3 -m vts_evaluation.make_queries \
        --labels-file "${labels_file}" \
        --out "${OUTPUT_ROOT}/${outdir}/queries.json"
    python3 -m vts_evaluation.evaluate_run \
        --graph "${main_graph}" \
        --node-gt "${OUTPUT_ROOT}/${outdir}/graph_0_node_gt.json" \
        --gt-trajectory "${gt_dir}" \
        --queries "${OUTPUT_ROOT}/${outdir}/queries.json" \
        --viz-path "${figures_dir}/topological_map.pdf" \
        > "${OUTPUT_ROOT}/${outdir}/metrics_report.json"

    python3 -m vts_evaluation.plot_lambda2 \
        --lambda2 "${OUTPUT_ROOT}/${outdir}/graph_0_lambda2.npy" \
        --valley-k 2.0 --skip 30 \
        --out "${figures_dir}/lambda2_valleys.pdf"
    python3 -m vts_evaluation.plot_odometry \
        --gt-trajectory "${gt_dir}" \
        --out "${figures_dir}/recorded_odometry.pdf"

    echo "=== ${config} | raw-map consistency check ==="
    python3 -m vts_evaluation.evaluate_run \
        --graph "${OUTPUT_ROOT}/${outdir}/graph_0_noopt.pkl" \
        --node-gt "${OUTPUT_ROOT}/${outdir}/graph_0_node_gt.json" \
        --gt-trajectory "${gt_dir}" \
        --no-viz \
        > "${OUTPUT_ROOT}/${outdir}/noopt_report.json"

    if [ "${RUN_ABLATIONS:-0}" = "1" ]; then
        for mode in visual geometric threshold; do
            echo "=== ${config} | ablation metrics: ${mode} ==="
            python3 -m vts_evaluation.evaluate_run \
                --graph "${OUTPUT_ROOT}/${base_outdir}_${mode}${VARIANT_SUFFIX}/graph_0_noopt.pkl" \
                --node-gt "${OUTPUT_ROOT}/${base_outdir}_${mode}${VARIANT_SUFFIX}/graph_0_node_gt.json" \
                --gt-trajectory "${gt_dir}" \
                --no-viz \
                > "${OUTPUT_ROOT}/${base_outdir}_${mode}${VARIANT_SUFFIX}/metrics_report.json"
        done
    fi
done

python3 -m vts_evaluation.summarize_experiments --root "${OUTPUT_ROOT}"
echo "All revised experiments done. Reports are in ${OUTPUT_ROOT}/*/."
