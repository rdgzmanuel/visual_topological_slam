from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

# ── Lab configurations ──────────────────────────────────────────────────────────

LAB_CONFIGS: dict[str, dict] = {
    "freiburg_a": {
        "start_1": (0.2, 0.0, 0.0),
        "start_2": (2.29, -0.29, 0.0),
        "trajectory_1": "cold-freiburg_part_a_seq2_night1",
        "trajectory_2": "cold-freiburg_part_a_seq2_sunny3",
        "world_limits": [-17.0, 19.75, -44.0, 16.5],
        "map_name": "freiburg_a.png",
        "origin": [521, 419],
        "weights": [
            0.0002033,
            0.005795,
            0.04014,
            -0.08563,
            1.196,
            -1.287,
            0.001895,
            0.01463,
            -0.03163,
            -0.2269,
            2.095,
            0.4223,
        ],
        "ext_rewiring": False,
    },
    "freiburg_ext": {
        "start_1": (0.29, 0.0, 0.0),
        "start_2": (0.46, -0.02, -0.11),
        "trajectory_1": "cold-freiburg_part_b_seq3_sunny1",
        "trajectory_2": "cold-freiburg_part_b_seq3_cloudy3",
        "world_limits": [-15.8, 18.8, -14.0, 37.0],
        "map_name": "freiburg_ext.png",
        "origin": [598, 1203],
        "weights": [
            -0.0002973,
            0.01101,
            -0.1336,
            0.4863,
            2.702,
            0.2607,
            0.00247,
            -0.001639,
            -0.1432,
            0.0684,
            3.786,
            -0.4859,
        ],
        "ext_rewiring": True,
    },
    "saarbruecken_a": {
        "start_1": (0.27, 0.03, 0.07),
        "start_2": (0.19, 0.01, 0.04),
        "trajectory_1": "cold-saarbruecken_part_a_seq2_night2",
        "trajectory_2": "cold-saarbruecken_part_a_seq2_cloudy1",
        "world_limits": [-16.75, 19.85, -37.5, 23.5],
        "map_name": "saarbruecken_a.png",
        "origin": [453, 580],
        "weights": [
            -0.0006797,
            0.006361,
            0.02416,
            -0.2676,
            1.739,
            0.6249,
            1.996e-05,
            0.0001553,
            -0.005155,
            -0.05405,
            1.929,
            0.4386,
        ],
        "ext_rewiring": True,
    },
    "saarbruecken_ext": {
        "start_1": (0.27, 0.0, 0.0),
        "start_2": (0.2, 0.0, 0.0),
        "trajectory_1": "cold-saarbruecken_part_b_seq4_sunny1",
        "trajectory_2": "cold-saarbruecken_part_b_seq4_cloudy1",
        "world_limits": [-11.7, 19.0, -24.0, 27.5],
        "map_name": "saarbruecken_ext.png",
        "origin": [430, 886],
        "weights": [
            -0.0005541,
            -0.0008833,
            0.03766,
            0.08971,
            2.01,
            0.8016,
            0.002923,
            -0.0005398,
            -0.1098,
            -0.1252,
            3.23,
            0.5326,
        ],
        "ext_rewiring": True,
    },
}

VALID_LABS: list[str] = list(LAB_CONFIGS.keys())
VALID_LOSSES: list[str] = ["contrastive", "triplet"]
VALID_MODES: list[str] = ["building", "command"]
VALID_COMMAND_MODES: list[str] = ["manual", "voice"]

ROS_LOG_ARGS: list[str] = ["--ros-args", "--log-level", "WARN"]


# ── Node builders ────────────────────────────────────────────────────────────────


def _build_building_nodes(
    lab: dict,
    model_name: str,
    trajectory_1: str,
    trajectory_2: str,
) -> list[Node]:
    return [
        Node(
            package="vts_graph_building",
            executable="graph_builder",
            output="screen",
            arguments=ROS_LOG_ARGS,
            parameters=[
                {
                    "start_1": lab["start_1"],
                    "start_2": lab["start_2"],
                    "world_limits": lab["world_limits"],
                    "map_name": lab["map_name"],
                    "origin": lab["origin"],
                    "weights": lab["weights"],
                    "trajectory_1": trajectory_1,
                    "trajectory_2": trajectory_2,
                    "ext_rewiring": lab["ext_rewiring"],
                    "model_name": model_name,
                    "publishing_topic": "graph_building_1",
                }
            ],
        ),
        Node(
            package="vts_camera",
            executable="camera",
            output="screen",
            arguments=ROS_LOG_ARGS,
            parameters=[
                {
                    "trajectory_1": trajectory_1,
                    "trajectory_2": trajectory_2,
                    "model_name": model_name,
                }
            ],
        ),
        Node(
            package="vts_map_alignment",
            executable="graph_alignment",
            output="screen",
            arguments=ROS_LOG_ARGS,
            parameters=[
                {
                    "trajectory": f"{trajectory_1}__{trajectory_2}",
                    "model_name": model_name,
                    "world_limits": lab["world_limits"],
                    "origin": lab["origin"],
                    "map_name": lab["map_name"],
                }
            ],
        ),
    ]


def _build_command_node(
    lab: dict,
    model_name: str,
    trajectory_1: str,
    trajectory_2: str,
    command_mode: str,
) -> list[Node]:
    return [
        Node(
            package="vts_commands",
            executable="commands",
            output="screen",
            arguments=ROS_LOG_ARGS,
            parameters=[
                {
                    "trajectory_1": trajectory_1,
                    "trajectory_2": trajectory_2,
                    "model_name": model_name,
                    "map_name": lab["map_name"],
                    "mode": command_mode,
                }
            ],
        ),
    ]


# ── Launch description ──────────────────────────────────────────────────────────


def generate_launch_description() -> LaunchDescription:
    import sys

    # Parse arguments (ros2 launch passes them as key:=value)
    def _get_arg(name: str, argv: list[str]) -> str | None:
        prefix: str = f"{name}:="
        for arg in argv:
            if arg.startswith(prefix):
                return arg[len(prefix) :]
        return None

    argv: list[str] = sys.argv

    loss: str = _get_arg("loss", argv) or "contrastive"
    lab_name: str = _get_arg("lab", argv) or "freiburg_a"
    mode: str = _get_arg("mode", argv) or "building"
    command_mode: str | None = _get_arg("command_mode", argv)

    # ── Validate ─────────────────────────────────────────────────────────────
    assert loss in VALID_LOSSES, f"'loss' must be one of {VALID_LOSSES}, got '{loss}'"
    assert lab_name in VALID_LABS, (
        f"'lab' must be one of {VALID_LABS}, got '{lab_name}'"
    )
    assert mode in VALID_MODES, f"'mode' must be one of {VALID_MODES}, got '{mode}'"
    if mode == "command":
        if command_mode is None:
            command_mode = "manual"
        assert command_mode in VALID_COMMAND_MODES, (
            f"'command_mode' must be one of {VALID_COMMAND_MODES}, got '{command_mode}'"
        )

    # ── Resolve config ───────────────────────────────────────────────────────
    lab: dict = LAB_CONFIGS[lab_name]
    model_name: str = f"visual_encoder_dino_{loss}_dim128_best"
    trajectory_1: str = lab["trajectory_1"]
    trajectory_2: str = lab["trajectory_2"]

    # ── Build nodes ──────────────────────────────────────────────────────────
    if mode == "building":
        nodes: list[Node] = _build_building_nodes(
            lab,
            model_name,
            trajectory_1,
            trajectory_2,
        )
    else:
        nodes = _build_command_node(
            lab,
            model_name,
            trajectory_1,
            trajectory_2,
            command_mode,
        )

    return LaunchDescription([
        DeclareLaunchArgument(
            "loss",
            default_value="contrastive",
            description="Loss type: contrastive | triplet",
        ),
        DeclareLaunchArgument(
            "lab",
            default_value="saarbruecken_a",
            description=f"Lab environment: {' | '.join(VALID_LABS)}",
        ),
        DeclareLaunchArgument(
            "mode", default_value="building", description="Mode: building | command"
        ),
        DeclareLaunchArgument(
            "command_mode",
            default_value="manual",
            description="Command mode: manual | voice (only used when mode=command)",
        ),
        *nodes,
    ])
