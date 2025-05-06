from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    model_name: str = "m13_ae_97"

    # lab: str = "freiburg_a"

    # start_1: tuple[float, float, float] = (0.2, 0.0, 0.0)
    # trajectory_1: str = "cold-freiburg_part_a_seq2_night1"

    # start_2: tuple[float, float, float] = (2.29, -0.29, 0.0)
    # trajectory_2: str = "cold-freiburg_part_a_seq2_sunny3"

    # lab: str = "freiburg_ext"
    # start_1: tuple[float, float, float] = (0.29, 0.0, 0.0)
    # trajectory_1: str = "cold-freiburg_part_b_seq3_sunny1"

    # start_2: tuple[float, float, float] = (0.46, -0.02, -0.11)
    # trajectory_2: str = "cold-freiburg_part_b_seq3_cloudy3"

    # lab: str = "saarbruecken_a"

    # start_1: tuple[float, float, float] = (0.27, 0.03, 0.07)
    # trajectory_1: str = "cold-saarbruecken_part_a_seq2_night2"

    # start_2: tuple[float, float, float] = (0.19, 0.01, 0.04)
    # trajectory_2: str = "cold-saarbruecken_part_a_seq2_cloudy1"

    lab: str = "saarbruecken_ext"
    start_1: tuple[float, float, float] = (0.2, 0.00, 0.00)
    trajectory_1: str = "cold-saarbruecken_part_b_seq4_cloudy1"

    start_2: tuple[float, float, float] = (0.27, 0.00, 0.00)
    trajectory_2: str = "cold-saarbruecken_part_b_seq4_sunny1"


    # lab: str = "ljubljana"
    # start: tuple[float, float, float] = (1.43, -5.61, 1.89)
    # trajectory: str = "cold-ljubljana_part_a_seq1_night1"

    settings: dict = {
        "freiburg_a": {
            "world_limits": (-17, 19.75, -44, 16.5),
            "map_name": "freiburg_a.png",
            "origin": (521, 419),
            "weights": (0.0002033, 0.005795, 0.04014, -0.08563, 1.196, -1.287,
                        0.001895, 0.01463, -0.03163, -0.2269, 2.095, 0.4223),
            "ext_rewiring": False
        },
        "saarbruecken_a": {
            "world_limits": (-16.75, 19.85, -37.5, 23.5),
            "map_name": "saarbruecken_a.png",
            "origin": (453, 580),
            "weights": (-0.0006797, 0.006361, 0.02416, -0.2676, 1.739, 0.6249,
                        1.996e-05, 0.0001553, -0.005155, -0.05405, 1.929, 0.4386),
            "ext_rewiring": False
        },
        "ljubljana": {
            "world_limits": (-6.15, 12.5, -7, 80),
            "map_name": "ljubljana.png",
            "origin": (270, 1574),
            "weights": (-0.0005541, -0.0008833, 0.03766, 0.08971, 2.01, 0.8016,
                        0.002923, -0.0005398, -0.1098, -0.1252, 3.23, 0.5326)
        },
        "saarbruecken_ext": {
            "world_limits": (-11.7, 19, -24, 27.5),
            "map_name": "saarbruecken_ext.png",
            "origin": (430, 886),
            "weights": (-0.0005541, -0.0008833, 0.03766, 0.08971, 2.01, 0.8016,
                        0.002923, -0.0005398, -0.1098, -0.1252, 3.23, 0.5326),
            "ext_rewiring": True
        },
        "freiburg_ext": {
            "world_limits": (-15.8, 18.8, -14, 37),
            "map_name": "freiburg_ext.png",
            "origin": (598, 1203),
            "weights": (-0.0002973, 0.01101, -0.1336, 0.4863, 2.702, 0.2607,
                        0.00247, -0.001639, -0.1432, 0.0684, 3.786, -0.4859),
            "ext_rewiring": True
        },
    }


    return LaunchDescription(
        [
            Node(
                package="vts_graph_building",
                executable="graph_builder",
                output="screen",
                arguments=["--ros-args", "--log-level", "WARN"],
                parameters=[{"start_1": start_1, "start_2": start_2, "world_limits": settings[lab]["world_limits"],
                             "map_name": settings[lab]["map_name"], "origin": settings[lab]["origin"],
                             "weights": settings[lab]["weights"], "trajectory_1": trajectory_1, "ext_rewiring": settings[lab]["ext_rewiring"],
                             "trajectory_2": trajectory_2, "model_name": model_name, "publishing_topic": "graph_building_1"}]),
            Node(
                package="vts_camera",
                executable="camera",
                output="screen",
                arguments=["--ros-args", "--log-level", "WARN"],
                parameters=[{"trajectory_1": trajectory_1, "trajectory_2": trajectory_2, "model_name": model_name}],
            ),
            Node(
                package="vts_map_alignment",
                executable="graph_alignment",
                output="screen",
                arguments=["--ros-args", "--log-level", "WARN"],
                parameters=[{"trajectory": f"{trajectory_1}__{trajectory_2}", "model_name": model_name,
                             "world_limits": settings[lab]["world_limits"], "origin": settings[lab]["origin"],
                             "map_name": settings[lab]["map_name"]}]),
        ]
    )




    # return LaunchDescription(
    #     [
    #         Node(
    #             package="vts_graph_building",
    #             executable="graph_builder",
    #             name="graph_builder_1",
    #             namespace="pass1",
    #             output="screen",
    #             arguments=["--ros-args", "--log-level", "WARN"],
    #             parameters=[{"start": start_1, "world_limits": settings[lab]["world_limits"],
    #                          "map_name": settings[lab]["map_name"], "origin": settings[lab]["origin"],
    #                          "weights": settings[lab]["weights"], "trajectory": trajectory_1,
    #                          "model_name": model_name, "publishing_topic": "graph_building_1"}],
    #             remappings=[("/camera", "/camera_1")] 
    #         ),
    #         Node(
    #             package="vts_graph_building",
    #             executable="graph_builder",
    #             name="graph_builder_2",
    #             namespace="pass2",
    #             output="screen",
    #             arguments=["--ros-args", "--log-level", "WARN"],
    #             parameters=[{"start": start_2, "world_limits": settings[lab]["world_limits"],
    #                          "map_name": settings[lab]["map_name"], "origin": settings[lab]["origin"],
    #                          "weights": settings[lab]["weights"], "trajectory": trajectory_2,
    #                          "model_name": model_name, "publishing_topic": "graph_building_2"}],
    #             remappings=[("/camera", "/camera_2")] 
    #         ),
    #         Node(
    #             package="vts_camera",
    #             executable="camera",
    #             name="camera_1",
    #             namespace="pass1",
    #             output="screen",
    #             arguments=["--ros-args", "--log-level", "WARN"],
    #             parameters=[{"trajectory": trajectory_1, "model_name": model_name}],
    #             remappings=[("/camera", "/camera_1")] 
    #         ),
    #         Node(
    #             package="vts_camera",
    #             executable="camera",
    #             name="camera_2",
    #             namespace="pass2",
    #             output="screen",
    #             arguments=["--ros-args", "--log-level", "WARN"],
    #             parameters=[{"trajectory": trajectory_2, "model_name": model_name}],
    #             remappings=[("/camera", "/camera_2")] 
    #         ),
    #         Node(
    #             package="vts_map_alignment",
    #             executable="graph_alignment",
    #             output="screen",
    #             arguments=["--ros-args", "--log-level", "WARN"],
    #             parameters=[{"trajectory": f"{trajectory_1}__{trajectory_2}", "model_name": model_name,
    #                          "world_limits": settings[lab]["world_limits"], "origin": settings[lab]["origin"]}]
    #         ),
    #     ]
    # )