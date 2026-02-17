"""ROS2 node for graph building in visual topological SLAM."""

from __future__ import annotations

import bisect
import gc
import os
import pickle
import sys
import time
from contextlib import suppress

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from vts_msgs.msg import FullGraph, GraphNode, ImageTensor

from vts_graph_building.graph_builder import GraphBuilder
from vts_graph_building.node import GraphNodeClass

# Type aliases
Pose2D = tuple[float, float, float]
Edge = tuple[GraphNodeClass, GraphNodeClass]


class GraphBuilderNode(Node):
    """ROS2 node for building topological graphs from visual input."""

    def __init__(self) -> None:
        """Initialize the GraphBuilderNode with parameters and subscriptions."""
        super().__init__("graph_builder")

        self._setup_communication()
        self._declare_and_load_parameters()
        self._initialize_state()

    def _declare_and_load_parameters(self) -> None:
        """Declare and load all ROS2 parameters."""
        # Window size
        self.declare_parameter("n", 30)
        self._n: int = self.get_parameter("n").get_parameter_value().integer_value

        # Peak detection parameters
        self.declare_parameter("gamma_proportion", 0.3)
        self._gamma_proportion: float = (
            self.get_parameter("gamma_proportion").get_parameter_value().double_value
        )

        self.declare_parameter("delta_proportion", 0.1)
        self._delta_proportion: float = (
            self.get_parameter("delta_proportion").get_parameter_value().double_value
        )

        # Distance threshold
        self.declare_parameter("distance_threshold", 1.5)
        self._distance_threshold: float = (
            self.get_parameter("distance_threshold").get_parameter_value().double_value
        )

        # Starting poses
        self.declare_parameter("start_1", (0.0, 0.0, 0.0))
        self._start_1: Pose2D = tuple(
            self
            .get_parameter("start_1")
            .get_parameter_value()
            .double_array_value.tolist()
        )

        self.declare_parameter("start_2", (0.0, 0.0, 0.0))
        self._start_2: Pose2D = tuple(
            self
            .get_parameter("start_2")
            .get_parameter_value()
            .double_array_value.tolist()
        )

        # World configuration
        self.declare_parameter("world_limits", (0.0, 0.0, 0.0, 0.0))
        self._world_limits: tuple[float, float, float, float] = tuple(
            self
            .get_parameter("world_limits")
            .get_parameter_value()
            .double_array_value.tolist()
        )

        # Map and trajectory names
        self.declare_parameter("map_name", "default_value")
        self._map_name: str = (
            self.get_parameter("map_name").get_parameter_value().string_value
        )

        self.declare_parameter("trajectory_1", "default_value")
        self._trajectory_1: str = (
            self.get_parameter("trajectory_1").get_parameter_value().string_value
        )

        self.declare_parameter("trajectory_2", "default_value")
        self._trajectory_2: str = (
            self.get_parameter("trajectory_2").get_parameter_value().string_value
        )

        # Model configuration
        self.declare_parameter("model_name", "default_value")
        self._model_name: str = (
            self.get_parameter("model_name").get_parameter_value().string_value
        )

        self.declare_parameter("ext_rewiring", False)
        self._ext_rewiring: bool = (
            self.get_parameter("ext_rewiring").get_parameter_value().bool_value
        )

        # Origin and weights
        self.declare_parameter("origin", (0, 0))
        self._origin: tuple[int, int] = tuple(
            self
            .get_parameter("origin")
            .get_parameter_value()
            .integer_array_value.tolist()
        )

        self.declare_parameter("weights", (0.0, 0.0, 0.0, 0.0))
        self._weights: tuple[float, ...] = tuple(
            self
            .get_parameter("weights")
            .get_parameter_value()
            .double_array_value.tolist()
        )

        # Uncomment if two maps already generated. See map_alignment_node.
        # self._publish_loaded_graphs()

    def _setup_communication(self) -> None:
        """Set up ROS2 subscriptions and publishers."""
        self._subscriber_camera = self.create_subscription(
            msg_type=ImageTensor,
            topic="/camera",
            callback=self._camera_callback,
            qos_profile=10,
        )

        self._graph_publisher = self.create_publisher(FullGraph, "/graph_alignment", 10)

    def _initialize_state(self) -> None:
        """Initialize node state variables."""
        self.graph_builder: GraphBuilder = self._create_graph_builder(
            trajectory=self._trajectory_1,
            start=self._start_1,
        )

        self._last_image_time: float = time.time()
        self._timeout_seconds: int = 20
        self._timer = self.create_timer(1.0, self._check_timeout)
        self._is_first_trajectory: bool = True
        self._valley_indices: list[int] = []
        self._prev_valley_index: int = 0

        # Odometry data (populated by _create_odometry_list if needed)
        self._timestamps: list[float] = []
        self._poses: list[Pose2D] = []

    def _create_odometry_list(self, trajectory: str) -> None:
        """
        Load odometry data from file for pose tracking.

        Args:
            trajectory: Name of the trajectory subfolder.
        """
        self._timestamps = []
        self._poses = []

        seq_data_folder: str = "/workspace/encoder/seq_data"
        odometry_file: str = os.path.join(
            seq_data_folder, trajectory, "odom_scans", "odom.tdf"
        )

        if not os.path.exists(odometry_file):
            self.get_logger().warn(f"Odometry file not found: {odometry_file}")
            return

        with open(odometry_file) as f:
            for line in f:
                parts: list[str] = line.strip().split()
                if len(parts) < 12:
                    continue

                t_sec: int = int(parts[3])
                t_usec: int = int(parts[4])
                timestamp: float = t_sec + t_usec * 1e-6

                x: float = float(parts[8])
                y: float = float(parts[9])
                theta: float = float(parts[11])

                self._timestamps.append(timestamp)
                self._poses.append((x, y, theta))

    def _create_graph_builder(self, trajectory: str, start: Pose2D) -> GraphBuilder:
        """
        Create a new GraphBuilder instance.

        Args:
            trajectory: Trajectory name.
            start: Starting pose.

        Returns:
            Configured GraphBuilder instance.
        """
        return GraphBuilder(
            n=self._n,
            gamma_proportion=self._gamma_proportion,
            delta_proportion=self._delta_proportion,
            distance_threshold=self._distance_threshold,
            initial_pose=start,
            world_limits=self._world_limits,
            map_name=self._map_name,
            origin=self._origin,
            weights=self._weights,
            trajectory=trajectory,
            model_name=self._model_name,
            ext_rewiring=self._ext_rewiring,
        )

    def _camera_callback(self, camera_msg: ImageTensor) -> None:
        """
        Process incoming camera messages.

        Args:
            camera_msg: Message containing image tensor and metadata.
        """
        self._last_image_time = time.time()

        image_name: str = camera_msg.image_name
        data: list[float] = camera_msg.data

        self.graph_builder.new_update_pose(image_name)
        array_data: np.ndarray = np.array(data, dtype=np.float32)
        self.graph_builder.update_matrices(array_data)

        if len(self.graph_builder.window_images) <= 1:
            return

        _, valley_idx = self.graph_builder.look_for_valley()

        if valley_idx not in (0, self._prev_valley_index - 1):
            self._valley_indices.append(valley_idx)
            self._prev_valley_index = valley_idx
            self.graph_builder.update_graph()
        elif len(self.graph_builder.graph) > 1:
            self.graph_builder.check_pose()

    def _get_closest_pose(self, query_time: float) -> Pose2D:
        """
        Find pose with closest timestamp to query time.

        Args:
            query_time: Target timestamp in seconds.

        Returns:
            Pose corresponding to closest timestamp.
        """
        if not self._timestamps:
            return (0.0, 0.0, 0.0)

        i: int = bisect.bisect_left(self._timestamps, query_time)

        if i == 0:
            return self._poses[0]
        if i >= len(self._timestamps):
            return self._poses[-1]

        before: float = self._timestamps[i - 1]
        after: float = self._timestamps[i]

        if abs(before - query_time) < abs(after - query_time):
            return self._poses[i - 1]
        return self._poses[i]

    def _publish_graph(self) -> None:
        """Publish graph completion message and save data."""
        graph_message: FullGraph = FullGraph()
        graph_message.edges = [1]

        self._plot_eigenvalues()
        self._save_graph_data(
            graph=self.graph_builder.graph,
            first=self._is_first_trajectory,
        )
        self.get_logger().warn("Graph saved")

        self._graph_publisher.publish(graph_message)
        self.get_logger().warn("Graph published")

    def _plot_eigenvalues(self) -> None:
        """Plot eigenvalues time series with valley markers."""
        max_plot_length: int = 800
        eigenvalues: list[float] = self.graph_builder.eigenvalues[:max_plot_length]
        indices: list[int] = [
            idx for idx in self._valley_indices if idx < max_plot_length
        ]

        output_file: str = "images/eigenvalues/eigenvalues.png"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(eigenvalues, color="lightblue", label="Eigenvalues")

        for idx in indices:
            plt.axvline(x=idx, color="darkblue", linestyle="--", linewidth=1)

        plt.title("Eigenvalues Time Series with Valley Indices")
        plt.xlabel("Image Index")
        plt.ylabel("Second Eigenvalue")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    def _save_graph_data(self, graph: list[Edge], first: bool) -> None:
        """
        Save graph data to pickle file.

        Args:
            graph: Graph edge list to save.
            first: Whether this is the first trajectory.
        """
        filename: str = "graph_1.pkl" if first else "graph_2.pkl"
        output_dir: str = f"graphs/{self._map_name[:-4]}"
        os.makedirs(output_dir, exist_ok=True)
        output_path: str = os.path.join(output_dir, filename)

        with open(output_path, "wb") as f:
            pickle.dump(graph, f)

        self.get_logger().warn(f"Graph saved to {output_path}")

    def _reset_graph_builder_for_second_trajectory(self) -> None:
        """Reset graph builder for processing second trajectory."""
        self.get_logger().warn(
            "First trajectory complete. Starting second trajectory..."
        )

        del self.graph_builder
        gc.collect()
        time.sleep(3)

        self.graph_builder = self._create_graph_builder(
            trajectory=self._trajectory_2,
            start=self._start_2,
        )
        self._is_first_trajectory = False
        self._valley_indices = []
        self._prev_valley_index = 0

        self.get_logger().warn("Reset successful")

    def _check_timeout(self) -> None:
        """Check for image reception timeout and handle accordingly."""
        if time.time() - self._last_image_time <= self._timeout_seconds:
            return

        if self._is_first_trajectory:
            self.get_logger().warn(
                "No images received. Generating map and starting second trajectory..."
            )
            self.graph_builder.generate_map()
            self._publish_graph()
            self._reset_graph_builder_for_second_trajectory()
            self._last_image_time = time.time()
            self.get_logger().warn("Ready for second trajectory.")
        else:
            self.get_logger().warn(
                "No images received. Generating map and shutting down..."
            )
            self.graph_builder.generate_map()
            self._publish_graph()
            time.sleep(3)
            sys.exit(0)

    def load_graph_data(self, filename: str) -> list[Edge]:
        """
        Load graph data from pickle file.

        Args:
            filename: Path to pickle file.

        Returns:
            Loaded graph edge list.

        Raises:
            FileNotFoundError: If file does not exist.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Graph data file '{filename}' not found.")

        with open(filename, "rb") as f:
            graph: list[Edge] = pickle.load(f)

        return graph

    def _publish_loaded_graphs(self) -> None:
        """Load and publish pre-computed graphs."""
        graph_names: list[str] = ["graph_1.pkl", "graph_2.pkl"]

        for graph_name in graph_names:
            graph_path: str = os.path.join("graphs", graph_name)

            try:
                graph: list[Edge] = self.load_graph_data(graph_path)
            except FileNotFoundError as e:
                self.get_logger().error(str(e))
                continue

            self.get_logger().warn(f"Publishing {graph_name}...")

            graph_list: list[GraphNode] = []
            edges: list[int] = []

            for node, adjacent in graph:
                node_message: GraphNode = GraphNode()
                node_message.pose = list(node.pose)
                node_message.shape = list(node.image.shape)
                node_message.image = node.image.flatten().tolist()
                node_message.features = node.visual_features.tolist()
                node_message.node_id = node.id

                graph_list.append(node_message)
                edges.append(node.id)
                edges.append(adjacent.id)

            graph_message: FullGraph = FullGraph()
            graph_message.nodes = graph_list
            graph_message.edges = edges

            self._graph_publisher.publish(graph_message)
            self.get_logger().warn("Graph published")


def main(args: list[str] | None = None) -> None:
    """
    Main entry point for the graph builder node.

    Args:
        args: Command line arguments.
    """
    rclpy.init(args=args)
    graph_builder_node: GraphBuilderNode = GraphBuilderNode()

    with suppress(KeyboardInterrupt):
        rclpy.spin(graph_builder_node)

    graph_builder_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
