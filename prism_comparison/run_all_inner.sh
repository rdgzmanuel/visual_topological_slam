#!/bin/bash
# Runs inside the PRISM-TopoMap container: all four COLD environments.
set -e
cd /workspace
echo "=== PRISM-TopoMap on COLD: starting all four environments ==="
for env in freiburg_a freiburg_ext saarbruecken_a saarbruecken_ext; do
    echo ""
    echo "=== Environment: $env ==="
    python3 driver/run_prism.py --env "$env" 2>&1 | tee "results/${env}_run.log" | grep -E "Device:|Loop|vertex at start|^\{|\"|Stopping" || true
done
echo ""
echo "=== All done. Results are in the results/ folder. ==="
