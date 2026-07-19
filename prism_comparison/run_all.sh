#!/bin/bash
# One-command launcher (Linux / macOS / WSL2).
# Tries the GPU first; falls back to CPU if the GPU run cannot start.
cd "$(dirname "$0")"
if docker compose run --rm prism; then
    echo "GPU run finished."
else
    echo ""
    echo "GPU run failed to start (no NVIDIA GPU or toolkit?). Falling back to CPU..."
    docker compose run --rm prism-cpu
fi
