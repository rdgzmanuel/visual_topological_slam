# Visual topological mapping with text-to-node retrieval

This repository contains a focused ROS 2 Humble pipeline that builds a
visual topological map from standard camera and odometry topics. It uses a
frozen, stock DINOv2 backbone: there is no training split, hard-negative
mining, fine-tuned checkpoint, or dataset-specific model state.

The mapping contribution consists of:

- spectral key-node detection over nonnegative visual affinities;
- odometry-first loop candidates, with bidirectional visual sequences used
  only to resolve geometrically ambiguous revisits, plus CLI-selectable
  ablations;
- timestamp-synchronized recorded wheel odometry with modeled uncertainty;
- distinct pose variables connected by uncertain odometry and loop factors;
- distinct observation poses without pose snapping or covariance reset;
- an optional GTSAM pose-graph ablation, disabled in the final method because
  it did not improve placement accuracy;
- loop-closure precision, recall and F1 evaluated separately from sequential
  edges.

Text-to-node retrieval is retained as a secondary feature. Representative
views stored at each node are indexed with CLIP and queried with text.

## Repository layout

```text
encoder/seq_data/              Raw COLD sequences used by the current player
vts_devcontainer/              ROS 2 development container
  vts_ws/src/vts_core/         Dataset-independent algorithms and metrics
  vts_ws/src/vts_players/      Dataset adapters (COLD and CID-SIMS)
  vts_ws/src/vts_mapping/      ROS graph-builder node
  vts_ws/src/vts_language/     Text-to-node retrieval node
  vts_ws/src/vts_evaluation/   Offline evaluation tools
  vts_ws/src/vts_bringup/      Launch file and experiment configurations
```

The mapper itself is dataset-agnostic. A new dataset or robot only needs to
publish the topic contract documented in
`vts_devcontainer/vts_ws/src/README.md`. CID-SIMS provides the independent
dataset evaluation without changing the mapper or its parameters.

## Build and run

Open the repository in its devcontainer, then run:

```bash
cd /workspaces/visual_topological_slam/vts_devcontainer/vts_ws
colcon build --symlink-install
source install/setup.bash

# One mapping run
ros2 launch vts_bringup pipeline.launch.py \
  config:=cold_freiburg_a.yaml mode:=building

# Complete fixed-parameter suite and gate ablations
./run_experiments.sh
```

### CID-SIMS

Download CID-SIMS from its official DOI
(`10.57760/sciencedb.ai.00003`) and extract the first traversal from each of
the three apartment scenes:

```text
encoder/
├── apartment1_1/groundtruth.txt
├── apartment2_1/groundtruth.txt
└── apartment3_1/groundtruth.txt
```

Then run the three independent-scene experiments:

```bash
./run_cid_sims_experiments.sh

# After the main result is validated, include all gate ablations:
RUN_ABLATIONS=1 ./run_cid_sims_experiments.sh
```

CID-SIMS has no room-level text annotations, so this suite evaluates mapping,
loop closures, odometry, storage and runtime but does not fabricate a language
benchmark from its object-segmentation masks.

Fresh outputs are written below `output/revised/`, deliberately separate from
graphs produced by the previous implementation. Use `FORCE=1
./run_experiments.sh` to rerun even when all source inputs are older than an
existing result. Use `SKIP_EXISTING=1 ./run_experiments.sh` to resume after a
runner-only fix without rebuilding completed graphs.

The experiment runner creates, for each environment:

- `final_graph.pkl` and `graph_0_noopt.pkl`;
- `metrics_report.json` and `noopt_report.json`;
- `graph_0_performance.json`, including DINOv2 inference time;
- visual-only, geometric-only, and threshold-baseline ablation reports.
- publication-size PDF maps, lambda₂ traces, and recorded-odometry plots in
  each main run's `figures/` directory.

At the end, `output/revised/summary.json` aggregates everything and the same
directory receives `mapping.csv`, `gate_ablation.csv`, `retrieval.csv`, and
`runtime.csv` for direct use when revising the paper.

The paper is intentionally updated only after these revised experiments have
completed.
