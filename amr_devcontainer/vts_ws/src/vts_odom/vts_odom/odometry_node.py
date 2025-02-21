import rclpy
import torch
import os
import time
from rclpy.node import Node

from nav_msgs.msg import Odometry
from vts_odom.odometry import Odometry

class OdometryNode(Node):
    def __init__(self) -> None:
        """
        Odometry node initializer.
        """
        super().__init__("odometry")
        self._publisher = self.create_publisher(Odometry, "/odom", 10)

        self._trajectory: str = "cold-freiburg_part_a_seq_1_cloudy1"
        self.odometry: Odometry = Odometry()

        self._publish_odometry()
    
    def _publish_odometry(self) -> None:
        """
        Function that publishes features extracted from images.
        """
        seq_data_folder: str = "/project/seq_data"
        odometry_folder: str = "odom_scans"
        odometry_file: str = "odom.tdf"
        trajectory_folder: str = os.path.join(seq_data_folder, self._trajectory)
        odometry_path: str = os.path.join(os.path.join(trajectory_folder, odometry_folder), odometry_file)

        with open(odometry_path, "r") as file:
            for i, line in enumerate(file):
                line_content: list[int | float] = line.split()
                if i == 0:
                    prev_timestamp: float = float(line_content[3] + "." + line_content[4])
                    prev_x: float = line_content[7]
                    prev_y: float = line_content[8]
                    prev_theta: float = line_content[10]
                    continue
                
                timestamp: float = float(line_content[3] + "." + line_content[4])
                x: float = line_content[7]
                y: float = line_content[8]
                theta: float = line_content[10]

                time_difference: float = timestamp - prev_timestamp
                v: float
                w: float
                v, w = self.odometry._compute_poses(time_difference, x, y, theta, prev_x, prev_y, prev_theta)
                odometry_msg: Odometry = Odometry()
                odometry_msg.twist.twist.linear.x = v
                odometry_msg.twist.twist.angular.z = w

                self._publisher.publish(odometry_msg)
                time.sleep(0.1)

                prev_timestamp = timestamp
                prev_x = x
                prev_y = y
                prev_theta = theta
        


def main(args=None):
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