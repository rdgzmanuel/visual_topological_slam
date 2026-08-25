"""Pipeline launch file.

All environment-specific values live in YAML configs under
``vts_bringup/config`` — the LAB_CONFIGS dictionary that used to be embedded
in the launch file is gone. Usage:

    ros2 launch vts_bringup pipeline.launch.py \
        config:=cold_freiburg_a.yaml mode:=building

Modes:
    building: player + graph builder (single-run; writes final_graph.pkl).
    command:  language node on an existing final_graph.pkl.

Ablation / experiment overrides (building mode):
    gate_mode:=both|visual|geometric|threshold
        Revisit-gate ablation. When a non-default mode is given, the mapping
        ``output_dir`` and ``run_name`` are suffixed with ``_<mode>`` so
        ablation runs never overwrite the main results.
    optimize:=true|false
        End-of-run SE(2) pose-graph optimization (default from the config;
        the pre-optimization graph is always saved as graph_*_noopt.pkl).
    dino_device:=auto|cuda|mps|cpu
        DINOv2 inference device. ``auto`` prefers CUDA, then Apple MPS, then
        CPU. MPS requires native macOS execution and is unavailable in the
        Linux Docker container used by this project.

Example gate ablation sweep for one environment:

    for m in both visual geometric threshold; do
        ros2 launch vts_bringup pipeline.launch.py \
            config:=cold_freiburg_a.yaml gate_mode:=$m
    done

The pipeline intentionally maps one traversal at a time; obsolete multi-map
alignment code has been removed from the focused implementation.
"""

from __future__ import annotations

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _make_nodes(context: object) -> list[Node]:
    config_name: str = LaunchConfiguration("config").perform(context)
    mode: str = LaunchConfiguration("mode").perform(context)
    gate_mode: str = LaunchConfiguration("gate_mode").perform(context)
    optimize: str = LaunchConfiguration("optimize").perform(context)
    dino_device: str = LaunchConfiguration("dino_device").perform(context)
    feature_backend: str = LaunchConfiguration("feature_backend").perform(context)
    visual_model: str = LaunchConfiguration("visual_model").perform(context)
    dino_model: str = LaunchConfiguration("dino_model").perform(context)
    dino_layer: str = LaunchConfiguration("dino_layer").perform(context)
    variant_suffix: str = LaunchConfiguration("variant_suffix").perform(context)

    config_path: str = os.path.join(
        get_package_share_directory("vts_bringup"), "config", config_name
    )
    with open(config_path) as f:
        config: dict[str, object] = yaml.safe_load(f)

    mapping: dict[str, object] = dict(config["mapping"])
    player: dict[str, object] = dict(config["player"])
    player_executable = str(config.get("player_executable", "cold_player"))
    if gate_mode:
        mapping["gate_mode"] = gate_mode
        if gate_mode != "both":
            # Keep ablation outputs separate from the main results.
            mapping["output_dir"] = f"{mapping['output_dir']}_{gate_mode}"
            mapping["run_name"] = f"{mapping['run_name']}_{gate_mode}"
    if optimize:
        mapping["optimize"] = optimize.strip().lower() in ("1", "true", "yes")
    if dino_device:
        mapping["dino_device"] = dino_device
    if feature_backend:
        mapping["feature_backend"] = feature_backend
    if visual_model:
        mapping["visual_model"] = visual_model
    if dino_model:
        mapping["dino_model"] = dino_model
    if dino_layer:
        mapping["dino_layer"] = int(dino_layer)
    if variant_suffix:
        mapping["output_dir"] = f"{mapping['output_dir']}{variant_suffix}"
        mapping["run_name"] = f"{mapping['run_name']}{variant_suffix}"

    nodes: list[Node] = []

    if mode == "building":
        dataset_player = Node(
            package="vts_players",
            executable=player_executable,
            name="dataset_player",
            parameters=[player],
            output="screen",
        )
        nodes.append(dataset_player)
        graph_builder: Node = Node(
            package="vts_mapping",
            executable="graph_builder",
            name="graph_builder",
            parameters=[mapping],
            output="screen",
        )
        nodes.append(graph_builder)
        # The graph builder exits by itself after the last sequence
        # (exit_when_done); take the player (and the launch) down with it so
        # scripted runs need no manual Ctrl-C.
        nodes.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=graph_builder,
                    on_exit=[
                        EmitEvent(
                            event=Shutdown(reason="graph builder finished")
                        )
                    ],
                )
            )
        )
        # A failed player cannot deliver the end-of-sequence message, so the
        # graph builder would otherwise wait forever. Shut the launch down and
        # let the runner reject the missing fresh graph.
        nodes.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=dataset_player,
                    on_exit=[
                        EmitEvent(event=Shutdown(reason="dataset player finished"))
                    ],
                )
            )
        )
    elif mode == "command":
        nodes.append(
            Node(
                package="vts_language",
                executable="commands",
                name="commands",
                parameters=[dict(config["language"])],
                output="screen",
            )
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value="cold_freiburg_a.yaml"),
            DeclareLaunchArgument("mode", default_value="building"),
            DeclareLaunchArgument(
                "gate_mode",
                default_value="",
                description=(
                    "Revisit-gate ablation: both|visual|geometric|threshold "
                    "(empty = use the config value)"
                ),
            ),
            DeclareLaunchArgument(
                "optimize",
                default_value="",
                description=(
                    "true|false: end-of-run pose-graph optimization "
                    "(empty = use the config value)"
                ),
            ),
            DeclareLaunchArgument(
                "dino_device",
                default_value="auto",
                description="DINOv2 inference device: auto|cuda|mps|cpu",
            ),
            DeclareLaunchArgument("feature_backend", default_value="dino_cls"),
            DeclareLaunchArgument("visual_model", default_value=""),
            DeclareLaunchArgument("dino_model", default_value=""),
            DeclareLaunchArgument("dino_layer", default_value=""),
            DeclareLaunchArgument("variant_suffix", default_value=""),
            OpaqueFunction(function=_make_nodes),
        ]
    )
