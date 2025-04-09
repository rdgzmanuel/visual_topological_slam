import rclpy
import sys
import pickle
import os
import numpy as np
import time
from rclpy.node import Node
from vts_msgs.msg import ImageTensor, GraphNode, FullGraph
from vts_graph_building.graph_builder import GraphBuilder


class GraphBuilderNode(Node):
    def __init__(self) -> None:
        """
        GraphBuilder node initializer.
        """
        super().__init__("graph_builder")

        # Parameters
        self.declare_parameter("n", 30)
        self._n: int = self.get_parameter("n").get_parameter_value().integer_value

        self.declare_parameter("gamma_proportion", 0.5) # lower bound for peaks
        self._gamma_proportion: float = self.get_parameter("gamma_proportion").get_parameter_value().double_value

        self.declare_parameter("delta_proportion", 0.11) # minimum difference of the a. c. between consecutive peaks
        self._delta_proportion: float = self.get_parameter("delta_proportion").get_parameter_value().double_value

        self.declare_parameter("distance_threshold", 3.0)
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

        self.declare_parameter("origin", (0, 0))
        self._origin: tuple[int, int] = tuple(self.get_parameter("origin").\
                                        get_parameter_value().integer_array_value.tolist())

        self.declare_parameter("weights", (0.0, 0.0, 0.0, 0.0))
        self._weights: tuple[float] = tuple(self.get_parameter("weights").\
                                      get_parameter_value().double_array_value.tolist())

        # self._subscriber_camera = self.create_subscription(
        #     msg_type=ImageTensor, topic="/camera", callback=self._camera_callback, qos_profile=10
        # )

        self._graph_publisher = self.create_publisher(FullGraph, "/graph_alignment", 10)

        # self.graph_builder: GraphBuilder = self._create_graph_builder(trajectory=self._trajectory_1,
        #                                                               start=self._start_1)

        self._last_image_time = time.time()
        self._timeout_seconds: int = 20

        # self._timer = self.create_timer(1.0, self._check_timeout)

        self._is_first_trajectory: bool = True


        self._publish_loaded_graphs()

        time.sleep(5)
        sys.exit(0)


    def _create_graph_builder(self, trajectory: str, start: tuple[float, float, float]) -> GraphBuilder:
        """
        Helper function to create a new instance of the GraphBuilder.
        """

        self.get_logger().warn("tryin to create builder")
        graph_builder = GraphBuilder(self._n, self._gamma_proportion, self._delta_proportion,
                            self._distance_threshold, start, self._world_limits, self._map_name,
                            self._origin, self._weights, trajectory, self._model_name)
        return graph_builder


    def _camera_callback(self, camera_msg: ImageTensor) -> None:
        """
        Callback function for the camera topic. Updates current node.

        Args:
            camera_msg (ImageTensor): Camera message containing tensor and shape data.
        """

        self.get_logger().warn("Receives message")
        self._last_image_time = time.time()

        prev_index: int = 0
        data: list[float] = camera_msg.data
        image_name: str = camera_msg.image_name

        self.get_logger().warn(f"name {image_name}")

        self.graph_builder.new_upgrade_pose(image_name)
        self.get_logger().warn("updates pose")
        array_data: np.ndarray = np.array(data).astype("float32")
        self.graph_builder.update_matrices(array_data)
        self.get_logger().warn("updates matrices")

        if len(self.graph_builder.window_images) > 1:
            lambda_2, valley_idx = self.graph_builder.look_for_valley()
            self.get_logger().warn("look for valley")

            if valley_idx not in [0, prev_index - 1]:
                if self.graph_builder.current_node is not None:
                    self.get_logger().warn(f"found valley: {self.graph_builder.current_node.pose}")
                else:
                    self.get_logger().warn("found valley")
                prev_index = valley_idx
                self.graph_builder.update_graph()
                self.get_logger().warn("Finished update")


    def _publish_graph(self) -> None:
        """
        Publishes graph in graph_bulding topic
        """
        self.get_logger().warn("Publishing Graph...")
        graph: list = []
        edges: list[int] = []

        for node, adjacent in self.graph_builder.graph:
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

            graph.append(node_message)
            edges.append(node.id)
            edges.append(adjacent.id)

        self.get_logger().warn("creating graph")

        graph_message: FullGraph = FullGraph()
        graph_message.nodes = graph
        graph_message.edges = edges

        self.save_graph_data(graph=self.graph_builder.graph, first=self._is_first_trajectory)
        self.get_logger().warn("Graph saved")

        self._graph_publisher.publish(graph_message)
        self.get_logger().warn("Graph published")


    def save_graph_data(self, graph: list[tuple[GraphNode, GraphNode]], first: bool):
        """
        Saves the graph and edges data to a file using pickle (binary format).
        """
        filename: str = "graph_1.pkl" if first else "graph_2.pkl"
        filename: str = os.path.join("graphs", filename)

        # Directly store the graph structure (including nodes and edges)
        # No need to manually serialize each field when using pickle
        with open(filename, "wb") as f:
            pickle.dump(graph, f)

        self.get_logger().warn("Graph saving done with pickle")
       

    def _reset_graph_builder_for_second_trajectory(self):
        """
        Resets the graph builder for processing the second trajectory by creating a new instance.
        """

        self.get_logger().warn("First trajectory complete. Starting to process the second trajectory.")

        new_graph_builder = self._create_graph_builder(self._trajectory_2, self._start_2)
        self.get_logger().warn("builder created")
        self.graph_builder = new_graph_builder
        self._is_first_trajectory = False

        self.get_logger().warn("Reset succesful")



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
                self._reset_graph_builder_for_second_trajectory()
                self._last_image_time = time.time()
                self.get_logger().warn("Ready to start processing second trajectory.")
            else:
                self.get_logger().warn("No images received for a while. Generating map and shutting down...")
                self.graph_builder.generate_map()
                self._publish_graph()
                return
    

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
