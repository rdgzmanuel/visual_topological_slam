import rclpy
import os
from rclpy.node import Node

from vts_msgs.msg import CustomOdometry
from geometry_msgs.msg import Quaternion
from vts_odom.odometry import OdometryClass

class OdometryNode(Node):
    def __init__(self) -> None:
        """
        Odometry node initializer.
        """
        super().__init__("odometry")
        self._publisher = self.create_publisher(CustomOdometry, "/odom", 10)

        self.declare_parameter("trajectory", "default_value")
        self._trajectory: str = self.get_parameter("trajectory").get_parameter_value().string_value
        self.odometry: OdometryClass = OdometryClass()

        self._publish_odometry()
    
    def _publish_odometry(self) -> None:
        """
        Reads the odometry file and starts a ROS2 timer to publish odometry messages.
        """
        seq_data_folder: str = "/workspace/project/seq_data"
        odometry_folder: str = "odom_scans"
        odometry_file: str = "odom.tdf"
        trajectory_folder: str = os.path.join(seq_data_folder, self._trajectory)
        odometry_path: str = os.path.join(trajectory_folder, odometry_folder, odometry_file)

        try:
            with open(odometry_path, "r") as file:
                lines: list[str] = file.readlines()
        except FileNotFoundError:
            self.get_logger().error(f"Odometry file not found: {odometry_path}")
            return

        if not lines:
            self.get_logger().error("Odometry file is empty.")
            return

        self.odom_data: list[list[float]] = [list(map(float, line.split())) for line in lines]
        self.odom_index: int = 0
        self.timer = self.create_timer(0.1, self._timer_callback)

    def _timer_callback(self) -> None:
        """
        Timer callback function that publishes odometry messages at regular intervals.
        """
        if self.odom_index >= len(self.odom_data):
            self.get_logger().info("Finished publishing odometry data.")
            self.timer.cancel()
            return

        line_content: list[float] = self.odom_data[self.odom_index]

        if self.odom_index == 0:
            self.prev_timestamp: float = float(f"{int(line_content[3])}.{int(line_content[4])}")
            self.prev_x: float = float(line_content[7])
            self.prev_y: float = float(line_content[8])
            self.prev_theta: float = float(line_content[11])
            self.odom_index += 1
            return

        timestamp: float = float(f"{int(line_content[3])}.{int(line_content[4])}")
        x: float = float(line_content[7])
        y: float = float(line_content[8])
        theta: float = float(line_content[11])

        time_difference: float = timestamp - self.prev_timestamp
        v, w = self.odometry._compute_poses(time_difference, x, y, theta, self.prev_x, self.prev_y, self.prev_theta)

        odometry_msg: CustomOdometry = CustomOdometry()
        odometry_msg.odometry.twist.twist.linear.x = v
        odometry_msg.odometry.twist.twist.angular.z = w
        odometry_msg.time_diff = time_difference

        # self._publisher.publish(odometry_msg)

        self.prev_timestamp = timestamp
        self.prev_x = x
        self.prev_y = y
        self.prev_theta = theta
        self.odom_index += 1


def main(args=None) -> None:
    """
    Main function to initialize and spin the ROS2 node.
    """
    rclpy.init(args=args)
    odometry_node: OdometryNode = OdometryNode()

    try:
        rclpy.spin(odometry_node)
    except KeyboardInterrupt:
        pass

    odometry_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
