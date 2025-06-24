import rclpy
import sys
import pickle
import os
import bisect
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
from rclpy.node import Node
from vts_msgs.msg import ImageTensor, GraphNode, FullGraph
from tf_transformations import euler_from_quaternion
from vts_graph_building.graph_builder import GraphBuilder
from typing import Optional


class GraphBuilderNode(Node):
    def __init__(self) -> None:
        """
        GraphBuilder node initializer.
        """
        super().__init__("graph_builder")

        # Parameters
        self.declare_parameter("n", 30)
        self._n: int = self.get_parameter("n").get_parameter_value().integer_value

        self.declare_parameter("gamma_proportion", 0.35) # lower bound for peaks 0.5   0.4 fE sE
        self._gamma_proportion: float = self.get_parameter("gamma_proportion").get_parameter_value().double_value

        self.declare_parameter("delta_proportion", 0.085) # minimum difference of the a. c. between consecutive peaks 0.11 / 0.09  fE sE
        self._delta_proportion: float = self.get_parameter("delta_proportion").get_parameter_value().double_value

        self.declare_parameter("distance_threshold", 1.0) # 3.5 fA   2.0 fE  4.0 sE  3.0 sA
        self._distance_threshold: float = self.get_parameter("distance_threshold").get_parameter_value().double_value

        self.declare_parameter("start_1", (0.0, 0.0, 0.0))
        self._start_1: tuple[float, float, float] = tuple(self.get_parameter("start_1").\
                                                  get_parameter_value().double_array_value.tolist())

        self.declare_parameter("start_2", (0.0, 0.0, 0.0))
        self._start_2: tuple[float, float, float] = tuple(self.get_parameter("start_2").\
                                                  get_parameter_value().double_array_value.tolist())

        self.declare_parameter("world_limits", (0.0, 0.0, 0.0, 0.0))
        self._world_limits: tuple[float, float, float, float] = tuple(self.get_parameter("world_limits").\
                                                                 get_parameter_value().double_array_value.tolist())

        self.declare_parameter("map_name", "default_value")
        self._map_name: str = self.get_parameter("map_name").get_parameter_value().string_value

        self.declare_parameter("trajectory_1", "default_value")
        self._trajectory_1: str = self.get_parameter("trajectory_1").get_parameter_value().string_value

        self.declare_parameter("trajectory_2", "default_value")
        self._trajectory_2: str = self.get_parameter("trajectory_2").get_parameter_value().string_value

        self.declare_parameter("model_name", "default_value")
        self._model_name: str = self.get_parameter("model_name").get_parameter_value().string_value

        self.declare_parameter("ext_rewiring", False)
        self._ext_rewiring: float = self.get_parameter("ext_rewiring").get_parameter_value().bool_value

        self.declare_parameter("origin", (0, 0))
        self._origin: tuple[int, int] = tuple(self.get_parameter("origin").\
                                        get_parameter_value().integer_array_value.tolist())

        self.declare_parameter("weights", (0.0, 0.0, 0.0, 0.0))
        self._weights: tuple[float] = tuple(self.get_parameter("weights").\
                                      get_parameter_value().double_array_value.tolist())

        self._subscriber_camera = self.create_subscription(
            msg_type=ImageTensor, topic="/camera", callback=self._camera_callback, qos_profile=10
        )

        self._graph_publisher = self.create_publisher(FullGraph, "/graph_alignment", 10)

        self.graph_builder: GraphBuilder = self._create_graph_builder(trajectory=self._trajectory_1,
                                                                      start=self._start_1,)

        self._timeout_seconds: int = 20
        self._last_image_time = time.time()

        self._timer = self.create_timer(1.0, self._check_timeout)

        self._is_first_trajectory: bool = True

        self._valley_indices: list[int] = []


        # Odometry
        self._odometry_path: str = "competition/odometry/odometry.csv"
        self._timestamps: list[float] = []
        self._poses: list[tuple[float, float, float]] = []
        self._current_timestamp: float = 0.0

        self._load_odometry_data()


    def _load_odometry_data(self) -> None:
        """
        Loads the odometry CSV and stores timestamp-sorted poses and their timestamps.
        """
        try:
            with open(self._odometry_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    timestamp = float(row["timestamp"])
                    # Extract position
                    x: float = float(row["pos_x"])
                    y: float = float(row["pos_y"])

                    # Extract orientation (quaternion)
                    qx: float = float(row["orient_x"])
                    qy: float = float(row["orient_y"])
                    qz: float = float(row["orient_z"])
                    qw: float = float(row["orient_w"])

                    _, _, theta = euler_from_quaternion([qx, qy, qz, qw])
                    self._timestamps.append(timestamp)
                    self._poses.append((x, y, theta))
        except Exception as e:
            self.get_logger().error(f"Error loading odometry data: {e}")


    def _create_graph_builder(self, trajectory: str, start: tuple[float, float, float]) -> GraphBuilder:
        """
        Helper function to create a new instance of the GraphBuilder.
        """

        # self.get_logger().warn("tryin to create builder")
        graph_builder = GraphBuilder(self._n, self._gamma_proportion, self._delta_proportion,
                            self._distance_threshold, start, self._world_limits, self._map_name,
                            self._origin, self._weights, trajectory, self._model_name, self._ext_rewiring)
        return graph_builder


    def _camera_callback(self, camera_msg: ImageTensor) -> None:
        """
        Callback function for the camera topic. Updates current node.

        Args:
            camera_msg (ImageTensor): Camera message containing tensor and shape data.
        """
        # self.get_logger().warn(f"image received")
        # self.graph_builder.plot_points()

        # x = [4, 4.29, 6.97, 27.35, 37.23, 39.84, 42.13]
        # y = [4, 2.5, 5.1, 28, 38, 39.84, 42.13]

        # poly = self.graph_builder.find_best_fit_function(x, y)
        # self.get_logger().warn(f" received")

        # self.graph_builder.draw_axes_with_scale_from_path("images/maps/competition.jpg")

        self._last_image_time = time.time()

        prev_index: int = 0
        data: list[float] = camera_msg.data
        image_name: str = camera_msg.image_name

        self._update_current_timestamp(image_name)

        pose = self._get_closest_pose(self._current_timestamp)

        self.graph_builder.update_pose(pose, image_name)

        array_data: np.ndarray = np.array(data).astype("float32")
        self.graph_builder.update_matrices(array_data)        

        if len(self.graph_builder.window_images) > 1:
            lambda_2, valley_idx = self.graph_builder.look_for_valley()

            if valley_idx not in [0, prev_index - 1]:
                self._valley_indices.append(valley_idx)
                prev_index = valley_idx
                self.graph_builder.update_graph()
            
            elif len(self.graph_builder.graph) > 1:
                self.graph_builder.check_pose()
    
        return None
    

    def _update_current_timestamp(self, image_name: str) -> None:
        """
        

        Args:
            image_name (str): _description_
        """
        self._current_timestamp = float(image_name[:-4].replace("_", "."))

        return None


    def _get_closest_pose(self, query_time: float) -> tuple[float, float, float]:
        """
        Finds the pose in the odometry buffer with the closest timestamp to the given query time.
        bisect_left(list, value) returns the index i where value should be inserted to
        maintain the list's sorted order.

        Args:
            query_time (float): The target timestamp (in seconds) for which to find the closest pose.

        Returns:
            tuple[float, float, float]: The (x, y, theta) pose corresponding to the closest timestamp.
        """
        i: int = bisect.bisect_left(self._timestamps, query_time)

        if i == 0:
            return self._poses[0]
        if i == len(self._timestamps):
            return self.poses[-1]

        before: float = self._timestamps[i - 1]
        after: float = self._timestamps[i]

        return self._poses[i - 1] if abs(before - query_time) < abs(after - query_time) else self._poses[i]


    def _publish_graph(self) -> None:
        """
        Publishes confirmation message in graph_bulding topic
        """
        graph_message: FullGraph = FullGraph()
        graph_message.edges = [1]

        self._plot_eigenvalues()

        self._save_graph_data(graph=self.graph_builder.graph, first=self._is_first_trajectory)
        self.get_logger().warn("Graph saved")

        self._graph_publisher.publish(graph_message)
        self.get_logger().warn("Graph published")
    

    def _plot_eigenvalues(self) -> None:
        """
        Plots the eigenvalues time series and highlights valley indices with vertical dashed lines.

        Output:
            A PNG image file at 'images/eigenvalues/eigenvalues.png'.
        """
        eigenvalues: list[float] = self.graph_builder.eigenvalues[:800]
        indices: list[int] = self._valley_indices

        cutoff_index = next((i for i, idx in enumerate(indices) if idx >= 800), len(indices))
        indices = indices[:max(cutoff_index - 1, 0)]

        output_file: str = "images/eigenvalues/eigenvalues.png"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.figure(figsize=(10, 6))
        # Plot eigenvalues time series
        plt.plot(eigenvalues, color='lightblue', label='Eigenvalues')
        # Plot vertical dashed lines at valley indices
        for idx in indices:
            plt.axvline(x=idx, color='darkblue', linestyle='--', linewidth=1)
        plt.title("Eigenvalues Time Series with Valley Indices")
        plt.xlabel("Image Index")
        plt.ylabel("Second Eigenvalue")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()


    def _save_graph_data(self, graph: list[tuple[GraphNode, GraphNode]], first: bool):
        """
        Saves the graph and edges data to a file using pickle (binary format).
        """
        filename: str = "graph_1.pkl" if first else "graph_2.pkl"
        output_path = os.path.join(f"graphs/{self._map_name[:-4]}", filename)

        with open(output_path, "wb") as f:
            pickle.dump(graph, f)

        self.get_logger().warn("Graph saving done with pickle")


    def _check_timeout(self) -> None:
        """
        Checks if the node has stopped receiving images. If no image is received
        within `self._timeout_seconds`, generate the map and shut down.
        """
        if time.time() - self._last_image_time > self._timeout_seconds:
            if self._is_first_trajectory:
                self.get_logger().warn("No images received for a while. Generating map and starting again...")
                
                self.graph_builder.generate_map()
                self._publish_graph()
                self._last_image_time = time.time()
                self.get_logger().warn("Ready to start processing second trajectory.")
            else:
                self.get_logger().warn("No images received for a while. Generating map and shutting down...")
                self.graph_builder.generate_map()
                self._publish_graph()
                time.sleep(3)
                sys.exit(0)
    

    def load_graph_data(self, filename: str) -> list[tuple[GraphNode, GraphNode]]:
        """
        Loads the graph and edges data from a pickle file.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Graph data file '{filename}' not found.")

        with open(filename, "rb") as f:
            graph = pickle.load(f)

        return graph
    

    def _publish_loaded_graphs(self) -> None:
        """
        
        """
        graph_names: list[str] = ["graph_1.pkl", "graph_2.pkl"]
        graphs = [self.load_graph_data(os.path.join("graphs", graph_name)) for graph_name in graph_names]

        for graph_ in graphs:
            self.get_logger().warn("Publishing Graph...")
            graph_list: list = []
            edges: list[int] = []

            for node, adjacent in graph_:
                # self.get_logger().warn("message")
                node_message: GraphNode = GraphNode()
                # self.get_logger().warn(f"pose {list(node.pose)}")
                node_message.pose = list(node.pose)
                # self.get_logger().warn(f"shape {list(node.image.shape)}")
                node_message.shape = list(node.image.shape)
                node_message.image = node.image.flatten().tolist()
                # self.get_logger().warn(f" visua shape {list(node.visual_features.shape)}")
                node_message.features = node.visual_features.tolist()
                node_message.node_id = node.id

                graph_list.append(node_message)
                edges.append(node.id)
                edges.append(adjacent.id)

            self.get_logger().warn("creating graph")

            graph_message: FullGraph = FullGraph()
            graph_message.nodes = graph_list
            graph_message.edges = edges

            self._graph_publisher.publish(graph_message)
            self.get_logger().warn("Graph published")


def main(args=None):
    rclpy.init(args=args)
    graph_builder_node = GraphBuilderNode()

    try:
        rclpy.spin(graph_builder_node)
    except KeyboardInterrupt:
        pass

    graph_builder_node.destroy_node()
    rclpy.try_shutdown()






if __name__ == "__main__":
    main()
