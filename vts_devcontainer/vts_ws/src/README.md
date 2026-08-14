# VTS — ROS 2 workspace

Dataset-agnostic implementation of the visual topological SLAM pipeline for
ROS 2 Humble (Python 3.10). The mapper never touches a dataset directly: it
consumes standard topics, and each dataset gets a small *player* node that
adapts it. See the repository-level `README.md` for the full setup and
reproduction guide.

## Packages

```
vts_core/        Pure-Python algorithm library (no ROS imports):
                 topo_graph, motion, features, node_detection, matching,
                 mapper, pose_graph, retrieval, metrics.
vts_players/     COLD player — the ONLY COLD-aware code in the project.
vts_mapping/     graph_builder node (image + odom -> final_graph.pkl).
vts_language/    commands node (natural-language query -> node target).
vts_evaluation/  Offline CLIs: evaluate_run, place_recognition_eval,
                 calibrate_floorplan, compare_odometry_maps, plot_odometry.
vts_viz/         Visualization helpers.
vts_bringup/     pipeline.launch.py + per-environment YAML configs.
vts_alignment/   Multi-map fusion (experimental; not part of the default
                 single-run flow and not launched).
```

## Topic contract

Any dataset player or real robot must provide:

| Topic                    | Type                      | Notes                     |
|--------------------------|---------------------------|---------------------------|
| `/camera/image`          | sensor_msgs/Image (bgr8)  |                           |
| `/odom`                  | nav_msgs/Odometry         | pose covariance filled    |
| `/ground_truth_pose`     | geometry_msgs/PoseStamped | optional, evaluation only |
| `/dataset/room_label`    | std_msgs/String           | optional, evaluation only |
| `/dataset/sequence_done` | std_msgs/String (JSON)    | end-of-run signal         |

The chi-square revisit gate relies on the covariance reported on `/odom`; the
COLD player fills it from the probabilistic motion model, and a real robot
must report a meaningful covariance as well.

## Quick start

```bash
cd vts_ws

colcon build --symlink-install
source install/setup.bash

# Build the map (single sequence -> output/<env>/final_graph.pkl):
ros2 launch vts_bringup pipeline.launch.py config:=cold_freiburg_a.yaml mode:=building

# Answer a natural-language query against the stored map:
ros2 launch vts_bringup pipeline.launch.py config:=cold_freiburg_a.yaml mode:=command

# Offline evaluation:
python3 -m vts_evaluation.evaluate_run \
    --graph output/freiburg_a/final_graph.pkl \
    --node-gt output/freiburg_a/graph_0_node_gt.json \
    --gt-trajectory /workspace/encoder/seq_data/cold-freiburg_part_a_seq2_night1/std_cam
```

Python dependencies beyond ROS: numpy, scipy, opencv-python, torch,
torchvision, transformers, pillow, pyyaml.
