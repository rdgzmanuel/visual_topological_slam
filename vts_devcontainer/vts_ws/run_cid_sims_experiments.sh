#!/usr/bin/env bash
# Run the same fixed-parameter mapper on three independent CID-SIMS scenes.
# Existing completed maps are reused by default, so an interrupted suite can
# be resumed safely. Set FORCE=1 to rebuild every map. After inspecting the
# three main results, use SKIP_EXISTING=1 RUN_ABLATIONS=1 to preserve them and
# compute only missing gate ablations.
set -euo pipefail

DATA_ROOT="${CID_SIMS_ROOT:-/workspace/encoder}"
OUTPUT_ROOT="output/revised"
VARIANT_SUFFIX="${VARIANT_SUFFIX:-}"
FEATURE_BACKEND="${FEATURE_BACKEND:-dino_cls}"
VISUAL_MODEL="${VISUAL_MODEL:-}"
DINO_MODEL="${DINO_MODEL:-}"
DINO_LAYER="${DINO_LAYER:-}"
VALLEY_MODE="${VALLEY_MODE:-}"
VALLEY_K="${VALLEY_K:-}"
VALLEY_DELTA="${VALLEY_DELTA:-}"

if { [ -n "${VALLEY_MODE}" ] || [ -n "${VALLEY_K}" ] || \
     [ -n "${VALLEY_DELTA}" ]; } && [ -z "${VARIANT_SUFFIX}" ]; then
    echo "Node-detector overrides require a non-empty VARIANT_SUFFIX." >&2
    echo "This safeguard prevents overwriting the final k=2 results." >&2
    exit 2
fi
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

wanted() {  # wanted <environment> <sequence> [filters...]
    local environment="$1"
    local sequence="$2"
    shift 2
    [ "$#" -eq 0 ] && return 0
    local filter
    for filter in "$@"; do
        if [ "${filter}" = "${environment}" ] || \
           [ "${filter}" = "${sequence}" ]; then
            return 0
        fi
    done
    return 1
}

for index in "${!SEQUENCES[@]}"; do
    sequence="${SEQUENCES[index]}"
    environment="${ENVIRONMENTS[index]}"
    wanted "${environment}" "${sequence}" "$@" || continue
    sequence_dir="${DATA_ROOT}/${sequence}"
    if [ ! -f "${sequence_dir}/groundtruth.txt" ] || \
       [ ! -f "${sequence_dir}/odom.txt" ] || \
       [ ! -d "${sequence_dir}/color" ]; then
        echo "CID-SIMS sequence not found or incomplete: ${sequence_dir}" >&2
        echo "Expected color/, groundtruth.txt and odom.txt." >&2
        exit 2
    fi
    python3 - "${sequence_dir}" <<'PY'
import os
from pathlib import Path
import sys

sequence_dir = Path(sys.argv[1])
images = sorted((sequence_dir / "color").glob("*.png"))
unavailable: list[Path] = []
for image in images:
    status = image.stat()
    if status.st_size == 0 or getattr(status, "st_blocks", 1) == 0:
        unavailable.append(image)
        if len(unavailable) == 5:
            break
if not images:
    raise SystemExit(f"No PNG images found in {sequence_dir / 'color'}")
if unavailable:
    examples = "\n  ".join(os.fspath(path) for path in unavailable)
    raise SystemExit(
        "CID-SIMS images are cloud-only placeholders and cannot be decoded.\n"
        "Download/keep the complete sequence on this device, then retry.\n"
        f"Examples:\n  {examples}"
    )
PY
done

MODES=(both)
if [ "${RUN_ABLATIONS:-0}" = "1" ]; then
    MODES+=(visual geometric threshold)
fi

for index in "${!CONFIGS[@]}"; do
    config="${CONFIGS[index]}"
    environment="${ENVIRONMENTS[index]}"
    base_environment="${environment}"
    environment="${environment}${VARIANT_SUFFIX}"
    sequence="${SEQUENCES[index]}"
    sequence_dir="${DATA_ROOT}/${sequence}"
    wanted "${base_environment}" "${sequence}" "$@" || continue

    # Preserve the already-computed GTSAM ablation before a final-method run
    # overwrites the standard artifact names. This is idempotent: once saved,
    # the comparison artifacts are never replaced by a non-GTSAM run.
    main_dir="${OUTPUT_ROOT}/${environment}"
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
            result_dir="${OUTPUT_ROOT}/${base_environment}_${mode}${VARIANT_SUFFIX}"
        fi
        result="${result_dir}/final_graph.pkl"
        stale=0
        for source in \
            "src/vts_core/vts_core/mapper.py" \
            "src/vts_core/vts_core/matching.py" \
            "src/vts_core/vts_core/topo_graph.py" \
            "src/vts_bringup/config/${config}"; do
            if [ -f "${result}" ] && [ "${source}" -nt "${result}" ]; then
                stale=1
            fi
        done
        if [ "${FORCE:-0}" != "1" ] && [ -f "${result}" ] && \
           { [ "${SKIP_EXISTING:-0}" = "1" ] || [ "${stale}" = "0" ]; }; then
            echo "=== ${config} | gate_mode=${mode} — existing result, skipping ==="
            continue
        fi
        echo "=== ${config} | gate_mode=${mode} ==="
        completion_marker="$(mktemp)"
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
        [ -z "${VALLEY_MODE}" ] || \
            launch_args+=(valley_mode:="${VALLEY_MODE}")
        [ -z "${VALLEY_K}" ] || launch_args+=(valley_k:="${VALLEY_K}")
        [ -z "${VALLEY_DELTA}" ] || \
            launch_args+=(valley_delta:="${VALLEY_DELTA}")
        ros2 launch vts_bringup pipeline.launch.py "${launch_args[@]}"
        if [ ! -f "${result}" ] || [ ! "${result}" -nt "${completion_marker}" ]; then
            rm -f "${completion_marker}"
            echo "Mapping failed: no fresh graph was produced at ${result}" >&2
            exit 3
        fi
        rm -f "${completion_marker}"
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
        --valley-mode "${VALLEY_MODE:-adaptive}" \
        --valley-k "${VALLEY_K:-2.0}" \
        --valley-delta "${VALLEY_DELTA:-0.1}" --skip 30 \
        --out "${figures_dir}/lambda2_valleys.pdf"
    python3 -m vts_evaluation.plot_odometry \
        --gt-trajectory "${gt_path}" \
        --odom "${sequence_dir}/odom.txt" \
        --out "${figures_dir}/recorded_odometry.pdf"

    if [ "${RUN_ABLATIONS:-0}" = "1" ]; then
        for mode in visual geometric threshold; do
            mode_dir="${OUTPUT_ROOT}/${base_environment}_${mode}${VARIANT_SUFFIX}"
            python3 -m vts_evaluation.evaluate_run \
                --graph "${mode_dir}/final_graph.pkl" \
                --node-gt "${mode_dir}/graph_0_node_gt.json" \
                --gt-trajectory "${gt_path}" --no-viz \
                > "${mode_dir}/metrics_report.json"
        done
    fi
done

if [ "$#" -eq 0 ] && [ -z "${VARIANT_SUFFIX}" ]; then
    python3 -m vts_evaluation.summarize_experiments --root "${OUTPUT_ROOT}"
    echo "All CID-SIMS experiments complete. Reports are under ${OUTPUT_ROOT}/cid_sims_apartment*/"
else
    echo "Selected or variant CID-SIMS experiments complete. Global summary preserved."
fi
