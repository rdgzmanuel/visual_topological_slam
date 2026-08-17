#!/usr/bin/env bash
# Run the same fixed-parameter mapper on three independent CID-SIMS scenes.
# Existing completed maps are reused by default, so an interrupted suite can
# be resumed safely. Set FORCE=1 to rebuild every map, and RUN_ABLATIONS=1
# only when the three main results have already been inspected.
set -euo pipefail

DATA_ROOT="${CID_SIMS_ROOT:-/workspace/encoder}"
OUTPUT_ROOT="output/revised"
CONFIGS=(
    cid_sims_apartment1_1.yaml
    cid_sims_apartment2_1.yaml
    cid_sims_apartment3_1.yaml
)
ENVIRONMENTS=(
    cid_sims_apartment1_1
    cid_sims_apartment2_1
    cid_sims_apartment3_1
)
SEQUENCES=(
    apartment1_1
    apartment2_1
    apartment3_1
)

for sequence in "${SEQUENCES[@]}"; do
    sequence_dir="${DATA_ROOT}/${sequence}"
    if [ ! -f "${sequence_dir}/groundtruth.txt" ] || \
       [ ! -f "${sequence_dir}/odom.txt" ] || \
       [ ! -d "${sequence_dir}/color" ]; then
        echo "CID-SIMS sequence not found or incomplete: ${sequence_dir}" >&2
        echo "Expected color/, groundtruth.txt and odom.txt." >&2
        exit 2
    fi
done

MODES=(both)
if [ "${RUN_ABLATIONS:-0}" = "1" ]; then
    MODES+=(visual geometric threshold)
fi

for index in "${!CONFIGS[@]}"; do
    config="${CONFIGS[index]}"
    environment="${ENVIRONMENTS[index]}"
    sequence="${SEQUENCES[index]}"
    sequence_dir="${DATA_ROOT}/${sequence}"

    # Preserve the already-computed GTSAM ablation before a final-method run
    # overwrites the standard artifact names. This is idempotent: once saved,
    # the comparison artifacts are never replaced by a non-GTSAM run.
    main_dir="${OUTPUT_ROOT}/${environment}"
    performance="${main_dir}/graph_0_performance.json"
    if [ -f "${main_dir}/metrics_report.json" ] && \
       [ ! -f "${main_dir}/visual_first_metrics_report.json" ]; then
        cp -p "${main_dir}/metrics_report.json" \
            "${main_dir}/visual_first_metrics_report.json"
    fi
    if [ -f "${main_dir}/metrics_report.json" ] && \
       [ ! -f "${main_dir}/always_visual_metrics_report.json" ]; then
        cp -p "${main_dir}/metrics_report.json" \
            "${main_dir}/always_visual_metrics_report.json"
    fi
    if [ -f "${performance}" ] && \
       grep -q '"optimizer_backend": "gtsam"' "${performance}"; then
        if [ ! -f "${main_dir}/gtsam_metrics_report.json" ] && \
           [ -f "${main_dir}/metrics_report.json" ]; then
            cp -p "${main_dir}/metrics_report.json" \
                "${main_dir}/gtsam_metrics_report.json"
        fi
        if [ ! -f "${main_dir}/gtsam_graph.pkl" ] && \
           [ -f "${main_dir}/final_graph.pkl" ]; then
            cp -p "${main_dir}/final_graph.pkl" \
                "${main_dir}/gtsam_graph.pkl"
        fi
        if [ ! -f "${main_dir}/gtsam_performance.json" ]; then
            cp -p "${performance}" "${main_dir}/gtsam_performance.json"
        fi
    fi

    for mode in "${MODES[@]}"; do
        if [ "${mode}" = "both" ]; then
            result_dir="${OUTPUT_ROOT}/${environment}"
        else
            result_dir="${OUTPUT_ROOT}/${environment}_${mode}"
        fi
        result="${result_dir}/final_graph.pkl"
        stale=0
        for source in \
            "src/vts_core/vts_core/mapper.py" \
            "src/vts_core/vts_core/matching.py" \
            "src/vts_bringup/config/${config}"; do
            if [ -f "${result}" ] && [ "${source}" -nt "${result}" ]; then
                stale=1
            fi
        done
        if [ "${FORCE:-0}" != "1" ] && [ -f "${result}" ] && [ "${stale}" = "0" ]; then
            echo "=== ${config} | gate_mode=${mode} — existing result, skipping ==="
            continue
        fi
        echo "=== ${config} | gate_mode=${mode} ==="
        ros2 launch vts_bringup pipeline.launch.py \
            config:="${config}" mode:=building gate_mode:="${mode}"
    done

    gt_path="${sequence_dir}/groundtruth.txt"
    figures_dir="${main_dir}/figures"
    mkdir -p "${figures_dir}"

    echo "=== ${config} | final no-GTSAM metrics ==="
    python3 -m vts_evaluation.evaluate_run \
        --graph "${main_dir}/final_graph.pkl" \
        --node-gt "${main_dir}/graph_0_node_gt.json" \
        --gt-trajectory "${gt_path}" \
        --viz-path "${figures_dir}/topological_map.pdf" \
        > "${main_dir}/metrics_report.json"

    echo "=== ${config} | raw-map consistency check ==="
    python3 -m vts_evaluation.evaluate_run \
        --graph "${main_dir}/graph_0_noopt.pkl" \
        --node-gt "${main_dir}/graph_0_node_gt.json" \
        --gt-trajectory "${gt_path}" --no-viz \
        > "${main_dir}/noopt_report.json"

    python3 -m vts_evaluation.plot_lambda2 \
        --lambda2 "${main_dir}/graph_0_lambda2.npy" \
        --valley-k 2.0 --skip 30 \
        --out "${figures_dir}/lambda2_valleys.pdf"
    python3 -m vts_evaluation.plot_odometry \
        --gt-trajectory "${gt_path}" \
        --odom "${sequence_dir}/odom.txt" \
        --out "${figures_dir}/recorded_odometry.pdf"

    if [ "${RUN_ABLATIONS:-0}" = "1" ]; then
        for mode in visual geometric threshold; do
            mode_dir="${OUTPUT_ROOT}/${environment}_${mode}"
            python3 -m vts_evaluation.evaluate_run \
                --graph "${mode_dir}/final_graph.pkl" \
                --node-gt "${mode_dir}/graph_0_node_gt.json" \
                --gt-trajectory "${gt_path}" --no-viz \
                > "${mode_dir}/metrics_report.json"
        done
    fi
done

python3 -m vts_evaluation.summarize_experiments --root "${OUTPUT_ROOT}"
echo "All CID-SIMS experiments complete. Reports are under ${OUTPUT_ROOT}/cid_sims_apartment*/"
