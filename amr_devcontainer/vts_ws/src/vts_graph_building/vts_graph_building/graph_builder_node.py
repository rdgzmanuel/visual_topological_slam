import rclpy
import numpy as np
import time
from rclpy.node import Node
from vts_msgs.msg import ImageTensor
from vts_graph_building.graph_builder import GraphBuilder


class GraphBuilderNode(Node):
    def __init__(self) -> None:
        """
        GraphBuilder node initializer.
        """
        super().__init__("graph_builder")

        # Parameters
        self.declare_parameter("n", 30)
        n: int = self.get_parameter("n").get_parameter_value().integer_value

        self.declare_parameter("gamma_proportion", 0.6)
        gamma_proportion: float = self.get_parameter("gamma_proportion").get_parameter_value().double_value

        self.declare_parameter("delta_proportion", 0.15)
        delta_proportion: float = self.get_parameter("delta_proportion").get_parameter_value().double_value

        self.declare_parameter("distance_threshold", 3.0)
        distance_threshold: float = self.get_parameter("distance_threshold").get_parameter_value().double_value

        self.declare_parameter("start", (0.0, 0.0, 0.0))
        start: tuple[float, float, float] = tuple(self.get_parameter("start").\
                                                  get_parameter_value().double_array_value.tolist())

        self.declare_parameter("world_limits", (0.0, 0.0, 0.0, 0.0))
        world_limits: tuple[float, float, float, float] = tuple(self.get_parameter("world_limits").\
                                                                 get_parameter_value().double_array_value.tolist())

        self.declare_parameter("map_name", "freiburg_a.png")
        map_name: str = self.get_parameter("map_name").get_parameter_value().string_value

        self.declare_parameter("trajectory", "freiburg_a.png")
        trajectory: str = self.get_parameter("trajectory").get_parameter_value().string_value

        self.declare_parameter("model_name", "freiburg_a.png")
        model_name: str = self.get_parameter("model_name").get_parameter_value().string_value

        self.declare_parameter("origin", (0, 0))
        origin: tuple[int, int] = tuple(self.get_parameter("origin").\
                                        get_parameter_value().integer_array_value.tolist())

        self.declare_parameter("weights", (0.0, 0.0, 0.0, 0.0))
        weights: tuple[float] = tuple(self.get_parameter("weights").\
                                      get_parameter_value().double_array_value.tolist())

        # subscription to the camera topic
        self._subscriber_camera = self.create_subscription(
            msg_type=ImageTensor, topic="/camera", callback=self._camera_callback, qos_profile=10
        )

        self.graph_builder: GraphBuilder = GraphBuilder(n, gamma_proportion, delta_proportion,
                                                        distance_threshold, start, world_limits, map_name,
                                                        origin, weights, trajectory, model_name)

        # timestamp tracking for last received image
        self._last_image_time = time.time()
        self._timeout_seconds = 5.0

        # timer to check for timeout
        self._timer = self.create_timer(1.0, self._check_timeout)

    def _camera_callback(self, camera_msg: ImageTensor) -> None:
        """
        Callback function for the camera topic. Updates current node.

        Args:
            camera_msg (ImageTensor): Camera message containing tensor and shape data.
        """
        self._last_image_time = time.time()  # Update timestamp when receiving an image

        prev_index: int = 0
        data: list[float] = camera_msg.data
        image_name: str = camera_msg.image_name

        self.graph_builder.new_upgrade_pose(image_name)
        array_data: np.ndarray = np.array(data).astype("float32")
        self.graph_builder.update_matrices(array_data)

        if len(self.graph_builder.window_images) > 1:
            lambda_2, valley_idx = self.graph_builder.look_for_valley()

            if valley_idx not in [0, prev_index - 1]:
                if self.graph_builder.current_node is not None:
                    self.get_logger().warn(f"found valley: {self.graph_builder.current_node.pose}")
                else:
                    self.get_logger().warn("found valley")
                prev_index = valley_idx
                self.graph_builder.update_graph()


    def _check_timeout(self) -> None:
        """
        Checks if the node has stopped receiving images. If no image is received
        within `self._timeout_seconds`, generate the map and shut down.
        """
        if time.time() - self._last_image_time > self._timeout_seconds:
            self.get_logger().info("No images received for a while. Generating map and shutting down...")
            self.graph_builder.generate_map()
            self.destroy_node()
            rclpy.shutdown()


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
