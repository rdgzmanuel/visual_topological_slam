#!/usr/bin/env bash
# Evaluate node-creation sensitivity from the saved lambda_2 traces.
#
# This script is offline and inexpensive: it does not launch ROS, rerun DINO,
# rebuild maps, or overwrite standard metrics reports. The constant-threshold
# baseline is calibrated only on Freiburg A and then frozen for every dataset.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/workspace/encoder/seq_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/revised}"
read -r -a k_values <<< "${K_VALUES:-1.0 1.5 2.0 2.5 3.0}"

case_args=()

add_cold_case() {
    local name="$1"
    local sequence="$2"
    local lambda2="${OUTPUT_ROOT}/${name}/graph_0_lambda2.npy"
    local sequence_root="${DATA_ROOT}/${sequence}"
    local images="${sequence_root}/std_cam"
    local labels="${sequence_root}/localization/places.lst"

    for required in "${lambda2}" "${images}" "${labels}"; do
        if [ ! -e "${required}" ]; then
            echo "Missing node-segmentation input: ${required}" >&2
            exit 2
        fi
    done
    case_args+=(--labeled-case "${name}" "${lambda2}" "${images}" "${labels}")
}

add_unlabeled_case() {
    local name="$1"
    local lambda2="${OUTPUT_ROOT}/${name}/graph_0_lambda2.npy"
    if [ ! -f "${lambda2}" ]; then
        echo "Missing node-segmentation input: ${lambda2}" >&2
        exit 2
    fi
    case_args+=(--case "${name}" "${lambda2}")
}

add_cold_case freiburg_a cold-freiburg_part_a_seq2_night1
add_cold_case freiburg_ext cold-freiburg_part_b_seq3_sunny1
add_cold_case saarbruecken_a cold-saarbruecken_part_a_seq2_night2
add_cold_case saarbruecken_ext cold-saarbruecken_part_b_seq4_sunny1
add_unlabeled_case cid_sims_apartment1_1
add_unlabeled_case cid_sims_apartment2_1
add_unlabeled_case cid_sims_apartment3_1

python3 -m vts_evaluation.node_segmentation \
    "${case_args[@]}" \
    --calibration-case freiburg_a \
    --k-values "${k_values[@]}" \
    --final-k 2.0 \
    --history 300 \
    --warmup 30 \
    --transition-tolerance-frames 15 \
    --frame-rate-hz 5.0 \
    --output-json "${OUTPUT_ROOT}/node_segmentation_experiment.json" \
    --output-csv "${OUTPUT_ROOT}/node_segmentation_experiment.csv"
