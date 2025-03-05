import rclpy
import numpy as np
import os
import time
from rclpy.node import Node
from torchvision import transforms
from PIL import Image
from node import GraphNode

from vts_msgs.msg import CustomOdometry
from vts_msgs.msg import ImageTensor
from vts_graph_building.graph_builder import GraphBuilder


class GraphBuilderNode(Node):
    def __init__(self) -> None:
        """
        GraphBuilder node initializer.
        """
        super().__init__("graph_builder")

        # Parameters
        self.declare_parameter("n", 50)
        n: int = self.get_parameter("n").get_parameter_value().integer_value

        self.declare_parameter("gamma_proportion", 0.8)
        gamma_proportion: float = self.get_parameter("gamma_proportion").get_parameter_value().double_value

        self.declare_parameter("delta_proportion", 0.8)
        delta_proportion: float = self.get_parameter("delta_proportion").get_parameter_value().double_value

        # Subscriptions
        self._subscriber_odom = self.create_subscription(
            msg_type=CustomOdometry, topic="/odom", callback=self._odom_callback, qos_profile=10
        )

        self._subscriber_camera = self.create_subscription(
            msg_type=ImageTensor, topic="/camera", callback=self._camera_callback, qos_profile=10
        )

        self.graph_builder: GraphBuilder = GraphBuilder(n, gamma_proportion, delta_proportion)
    
    def _odom_callback(self, odom_msg: CustomOdometry) -> None:
        """
        Callback function for the odom topic. Updates current pose.

        Args:
            odom_msg (CustomOdometry): custom odometry message. Contains lineal and
            angular velocities, as well as time difference between different messages.
        """
        v: float = odom_msg.odometry.twist.twist.lineal.x
        w: float = odom_msg.odometry.twist.twist.angular.z
        time_difference: float = odom_msg.time_diff

        self.graph_builder.update_pose(v, w, time_difference)

        return None


    def _camera_callback(self, camera_msg: ImageTensor) -> None:
        """
        Callback function for the camera topic. Updates current node.

        Args:
            camera_msg (ImageTensor): camera message. Contains tensor and shape data.
        """
        
        shape: list[int] = camera_msg.shape
        data: list[float] = camera_msg.data

        array_data: np.ndarray = np.array(data).astype("float32")

        self.graph_builder.update_matrices(array_data)

        if self.graph_builder.window_full:
            found: bool
            valley_idx: int
            found, valley_idx = self.graph_builder.look_for_valley()

            if found:
                self.graph_builder.update_graph(valley_idx)

        return None


def main(args=None):
    rclpy.init(args=args)
    graph_builder_node: GraphBuilderNode = GraphBuilderNode()

    try:
        rclpy.spin(graph_builder_node)
    except KeyboardInterrupt:
        pass

    graph_builder_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()