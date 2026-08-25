#!/usr/bin/env bash
# AnyLoc-GeM literature baseline using the same mapper and gate settings.
#
# Default: computationally matched DINOv2 ViT-S/14, last-block value facets,
# GeM p=3. Report this as "AnyLoc-GeM (ViT-S adaptation)".
#
# Exact published encoder (very large; CUDA strongly recommended):
#   ANYLOC_EXACT=1 ./run_anyloc_baseline.sh
set -euo pipefail

if [ "${ANYLOC_EXACT:-0}" = "1" ]; then
    export DINO_MODEL="dinov2_vitg14"
    export DINO_LAYER="31"
    export VARIANT_SUFFIX="_anyloc_gem_vitg"
else
    export DINO_MODEL="${DINO_MODEL:-dinov2_vits14}"
    export DINO_LAYER="${DINO_LAYER:-11}"
    export VARIANT_SUFFIX="${VARIANT_SUFFIX:-_anyloc_gem_vits}"
fi
export FEATURE_BACKEND="anyloc_gem"
export VISUAL_MODEL="cosine"
export RUN_ABLATIONS=0

suite="${1:-all}"
if [ "$#" -gt 0 ]; then shift; fi

case "$suite" in
    all)
        ./run_experiments.sh "$@"
        ./run_cid_sims_experiments.sh "$@"
        ;;
    cold) ./run_experiments.sh "$@" ;;
    cid-sims) ./run_cid_sims_experiments.sh "$@" ;;
    *)
        echo "Usage: $0 [all|cold|cid-sims] [environment ...]" >&2
        exit 2
        ;;
esac
