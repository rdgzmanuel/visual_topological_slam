#!/usr/bin/env bash
# Re-score completed maps at several ground-truth distance tolerances.
#
# This script does not launch ROS, extract features, rebuild graphs, or modify
# the standard metrics_report.json files. By default it evaluates the seven
# final no-GTSAM maps and the AnyLoc-GeM ViT-S baseline at 0.5, 1, 2, and 3
# metres. Set INCLUDE_ABLATIONS=1 to score the existing
# visual/geometric/threshold gate ablations as well. Set INCLUDE_ANYLOC=0 only
# when the baseline artifacts are intentionally unavailable.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/encoder/seq_data}"
CID_SIMS_ROOT="${CID_SIMS_ROOT:-/workspace/encoder}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/revised}"
read -r -a tolerances <<< "${TOLERANCES:-0.5 1.0 2.0 3.0}"

cold_environments=(
    freiburg_a
    freiburg_ext
    saarbruecken_a
    saarbruecken_ext
)
cold_sequences=(
    cold-freiburg_part_a_seq2_night1
    cold-freiburg_part_b_seq3_sunny1
    cold-saarbruecken_part_a_seq2_night2
    cold-saarbruecken_part_b_seq4_sunny1
)
cid_environments=(
    cid_sims_apartment1_1
    cid_sims_apartment2_1
    cid_sims_apartment3_1
)
cid_sequences=(
    apartment1_1
    apartment2_1
    apartment3_1
)

case_args=()

add_case() {
    local name="$1"
    local result_dir="$2"
    local gt_path="$3"
    local graph="${result_dir}/graph_0_noopt.pkl"
    local node_gt="${result_dir}/graph_0_node_gt.json"

    for required in "${graph}" "${node_gt}" "${gt_path}"; do
        if [ ! -e "${required}" ]; then
            echo "Missing tolerance-sweep input: ${required}" >&2
            echo "Complete the corresponding mapping run before retrying." >&2
            exit 2
        fi
    done
    case_args+=(--case "${name}" "${graph}" "${node_gt}" "${gt_path}")
}

add_environment() {
    local environment="$1"
    local gt_path="$2"
    add_case "${environment}" "${OUTPUT_ROOT}/${environment}" "${gt_path}"

    if [ "${INCLUDE_ANYLOC:-1}" = "1" ]; then
        add_case \
            "${environment}_anyloc_gem_vits" \
            "${OUTPUT_ROOT}/${environment}_anyloc_gem_vits" \
            "${gt_path}"
    fi

    if [ "${INCLUDE_ABLATIONS:-0}" = "1" ]; then
        local mode
        for mode in visual geometric threshold; do
            add_case \
                "${environment}_${mode}" \
                "${OUTPUT_ROOT}/${environment}_${mode}" \
                "${gt_path}"
        done
    fi
}

for index in "${!cold_environments[@]}"; do
    environment="${cold_environments[index]}"
    sequence="${cold_sequences[index]}"
    add_environment \
        "${environment}" \
        "${DATA_ROOT}/${sequence}/std_cam"
done

for index in "${!cid_environments[@]}"; do
    environment="${cid_environments[index]}"
    sequence="${cid_sequences[index]}"
    add_environment \
        "${environment}" \
        "${CID_SIMS_ROOT}/${sequence}/groundtruth.txt"
done

python3 -m vts_evaluation.tolerance_sweep \
    --tolerances "${tolerances[@]}" \
    "${case_args[@]}" \
    --output-json "${OUTPUT_ROOT}/loop_tolerance_sweep.json" \
    --output-csv "${OUTPUT_ROOT}/loop_tolerance_sweep.csv"
