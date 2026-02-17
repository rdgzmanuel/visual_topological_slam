# Visual Topological SLAM using Deep Learning Techniques

A system for constructing topological maps of indoor environments using odometry measurements and camera images. Visual inputs are processed through a deep learning encoder to extract feature representations used for similarity-based node extraction. The system includes loop closure detection, trajectory fusion, semantic object annotation, and voice-controlled navigation.

**Author:** Manuel Rodriguez Villegas
**University:** Comillas Pontifical University
**Year:** 2026

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Encoder: Model Training and Evaluation](#encoder-model-training-and-evaluation)
  - [Configuration](#configuration)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [ROS 2 System: Topological SLAM](#ros-2-system-topological-slam)
  - [Building the Workspace](#building-the-workspace)
  - [Running Experiments](#running-experiments)
  - [Launch Arguments](#launch-arguments)
- [Citation](#citation)

---

## Overview

Topological maps represent an environment as a graph, where nodes correspond to points of interest and edges represent traversable paths. Compared to metric maps, they are computationally lighter and integrate naturally with natural language modules.

This project implements the following pipeline:

1. A deep learning encoder extracts visual feature representations from camera images.
2. Similarity comparisons between feature vectors drive node extraction during exploration.
3. Odometry data refines node positions for added robustness.
4. A loop closure detection and rewiring mechanism updates edges when a closure is detected.
5. Two trajectories are aligned and their nodes fused to correct detection errors.
6. Semantic information is incorporated by detecting objects in each node's image, describing them in natural language, and embedding the resulting sentence with a language model, enabling voice-controlled navigation.

---

## Repository Structure

```
visual_topological_slam/
├── encoder/              # Deep learning model training and evaluation
│   ├── src/
│   │   └── train.py      # Training entry point
│   ├── config.py          # Hyperparameters and loss function selection
│   └── evaluate.py        # Model evaluation
├── vts_devcontainer/      # ROS 2 dev container
│   └── vts_ws/            # ROS 2 workspace
│       └── launch/
│           └── project.launch.py
├── docs/                  # Documentation and GitHub Pages assets
├── thesis/                # Thesis document
├── Dockerfile
└── requirements.txt
```

---

## Encoder: Model Training and Evaluation

The `encoder/` directory contains everything needed to train and evaluate the visual feature extraction model.

### Configuration

Edit `encoder/config.py` to set hyperparameters and select the loss function. The two supported loss functions are:

- **Triplet loss** -- learns an embedding space where anchor-positive distances are smaller than anchor-negative distances by a margin.
- **Contrastive loss** -- pulls similar pairs together and pushes dissimilar pairs apart in the embedding space.

Adjust parameters such as learning rate, batch size, number of epochs, and other training settings directly in `config.py`.

### Training

From the `encoder/` directory, run:

```bash
python -m src.train
```

### Evaluation

After training, evaluate the model with:

```bash
python -m src.evaluate
```

---

## ROS 2 System: Topological SLAM

The `vts_devcontainer/` directory contains the full ROS 2 project for running topological SLAM experiments.

### Building the Workspace

Navigate to the ROS 2 workspace and build:

```bash
cd vts_devcontainer/vts_ws
colcon build --symlink-install
source install/setup.bash
```

> **Note:** You need to source the workspace in every new terminal session, or add the source command to your shell configuration.

### Running Experiments

Launch the full system with:

```bash
ros2 launch launch/project.launch.py
```

### Launch Arguments

The launch file accepts the following arguments to configure experiments:

| Argument | Options | Description |
|---|---|---|
| `loss` | `triplet`, `contrastive` | Loss function used by the encoder model. |
| `lab` | `freiburg_a`, `freiburg_ext`, `saarbruecken_a`, `saarbruecken_ext` | Laboratory / dataset environment to use. |
| `mode` | `building`, `command` | Operation mode. `building` constructs the topological map; `command` enables navigation. |
| `command_mode` | `manual`, `voice` | Only applicable when `mode:=command`. Selects manual or voice-controlled navigation. |

Example usage:

```bash
ros2 launch launch/project.launch.py loss:=triplet lab:=freiburg_a mode:=command command_mode:=voice
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@thesis{rodriguez2025vts,
  title   = {Visual Topological SLAM using Deep Learning Techniques},
  author  = {Rodriguez Villegas, Manuel},
  school  = {Comillas Pontifical University},
  year    = {2026}
}
```