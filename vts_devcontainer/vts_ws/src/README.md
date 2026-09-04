# TopoSIGMA — ROS 2 workspace

The workspace targets ROS 2 Jazzy on Ubuntu 24.04 and CPython 3.12. Its
third-party Python environment is managed centrally by uv from
`../../pyproject.toml` and `../../uv.lock`.

TopoSIGMA consumes standard ROS topics and never accesses a dataset
directly. Each dataset has a small player that implements the same interface.
The COLD player synchronizes camera frames to the dataset's recorded wheel
odometry. Because COLD provides no covariance, one fixed motion model
propagates uncertainty without perturbing the measured trajectory.

## Packages

```
vts_core/        Pure-Python algorithm library (no ROS imports):
                 topo_graph, motion, features, directional, node_detection,
                 matching, mapper, pose_graph, retrieval, metrics.
vts_players/     COLD player — the ONLY COLD-aware code in the project.
vts_mapping/     graph_builder node (image + odom -> final_graph.pkl).
vts_language/    commands node (natural-language query -> node target).
vts_evaluation/  Offline metrics, retrieval evaluation and map plots.
vts_bringup/     pipeline.launch.py + per-environment YAML configs.
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

The revisit mechanism uses covariance reported on `/odom` to generate and
rank geometrically compatible candidates. A unique candidate is accepted
directly; when several places remain plausible, bidirectional three-node
sequences compare training-free vMF distributions fitted to each node's
normalized DINO descriptors. A real robot or another player must therefore
report meaningful covariance.

## Quick start

```bash
cd /workspaces/topo_sigma/vts_devcontainer/vts_ws

colcon build --symlink-install
source install/setup.bash

# Build the map (single sequence -> output/revised/<env>/final_graph.pkl):
ros2 launch vts_bringup pipeline.launch.py config:=cold_freiburg_a.yaml mode:=building

# Answer a natural-language query against the stored map:
ros2 launch vts_bringup pipeline.launch.py config:=cold_freiburg_a.yaml mode:=command

# Offline evaluation:
python3 -m vts_evaluation.evaluate_run \
    --graph output/revised/freiburg_a/final_graph.pkl \
    --node-gt output/revised/freiburg_a/graph_0_node_gt.json \
    --gt-trajectory /workspace/encoder/seq_data/cold-freiburg_part_a_seq2_night1/std_cam
```

Python dependencies beyond ROS: numpy, scipy, opencv-python, torch,
torchvision, transformers, pillow, pyyaml.

## DINOv2 inference device

The graph builder defaults to `dino_device:=auto`, which selects CUDA, then
Apple MPS, then CPU according to what the installed PyTorch runtime exposes.
The selected device is printed once when the graph builder starts. It can be
overridden for diagnosis or reproducibility:

```bash
ros2 launch vts_bringup pipeline.launch.py \
    config:=cold_freiburg_a.yaml dino_device:=cpu
```

Valid values are `auto`, `cuda`, `mps`, and `cpu`; explicitly requesting an
unavailable accelerator fails immediately. Apple Metal/MPS is not exposed to
Linux containers by Docker Desktop, so this project's devcontainer uses CPU
on a Mac. MPS is available only when running the Python pipeline natively on
macOS with an MPS-enabled PyTorch installation. On a CUDA workstation, Docker
must be configured with the NVIDIA Container Toolkit, the container must be
started with GPU access, and PyTorch must be a CUDA-enabled build. Device
selection cannot provide those host/container capabilities by itself.
