"""Pipeline launch file.

All environment-specific values live in YAML configs under
``vts_bringup/config`` — the LAB_CONFIGS dictionary that used to be embedded
in the launch file is gone. Usage:

    ros2 launch vts_bringup pipeline.launch.py \
        config:=cold_freiburg_a.yaml mode:=building

Modes:
    building: player + graph builder + alignment.
    command:  language node on an existing final_graph.pkl.
"""

from __future__ import annotations

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _make_nodes(context: object) -> list[Node]:
    config_name: str = LaunchConfiguration("config").perform(context)
    mode: str = LaunchConfiguration("mode").perform(context)

    config_path: str = os.path.join(
        get_package_share_directory("vts_bringup"), "config", config_name
    )
    with open(config_path) as f:
        config: dict[str, object] = yaml.safe_load(f)

    nodes: list[Node] = []

    if mode == "building":
        nodes.append(
            Node(
                package="vts_players",
                executable="cold_player",
                name="cold_player",
                parameters=[dict(config["player"])],
                output="screen",
            )
        )
        nodes.append(
            Node(
                package="vts_mapping",
                executable="graph_builder",
                name="graph_builder",
                parameters=[dict(config["mapping"])],
                output="screen",
            )
        )
        nodes.append(
            Node(
                package="vts_alignment",
                executable="map_alignment",
                name="map_alignment",
                parameters=[dict(config["alignment"])],
                output="screen",
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
            OpaqueFunction(function=_make_nodes),
        ]
    )
