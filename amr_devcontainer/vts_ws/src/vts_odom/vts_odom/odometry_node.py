import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Time
import csv
import os


class PosePublisher(Node):
    """
    A ROS 2 node that publishes geometry_msgs/PoseStamped messages at fixed intervals
    from a CSV file containing timestamped pose data.
    """

    def __init__(self) -> None:
        """
        Initializes the PosePublisher node.

        - Declares and retrieves the path to the odometry CSV file.
        - Reads the CSV file into memory.
        - Creates a ROS 2 timer that triggers at 0.2-second intervals.
        """
        super().__init__("pose_publisher")
        self.publisher_ = self.create_publisher(PoseStamped, "/pose", 10)

        self.declare_parameter("odometry_path", "topological_map/odometry/odometry.csv")
        odometry_path: str = self.get_parameter("odometry_path").get_parameter_value().string_value

        self.data: list[dict[str, str]] = []
        self.index: int = 0
        self.timer: Timer | None = None
        self.pub_time: float = 0.2

        if not os.path.exists(odometry_path):
            self.get_logger().error(f"Odometry file not found: {odometry_path}")
            return

        try:
            with open(odometry_path, "r") as csvfile:
                reader = csv.DictReader(csvfile)
                self.data = list(reader)
        except Exception as e:
            self.get_logger().error(f"Error reading odometry CSV: {e}")
            return

        if not self.data:
            self.get_logger().error("Odometry CSV is empty or has invalid format.")
            return

        self.timer = self.create_timer(self.pub_time, self.timer_callback)


    def timer_callback(self) -> None:
        """
        Timer callback function that publishes a PoseStamped message.

        Uses the next line of pose data from the loaded CSV file and converts it
        into a ROS 2 PoseStamped message with the appropriate timestamp and frame.
        """
        if self.index >= len(self.data):
            self.get_logger().info("Finished publishing all pose data.")
            if self.timer:
                self.timer.cancel()
            return

        row: dict[str, str] = self.data[self.index]

        try:
            timestamp: float = float(row['timestamp'])
            msg: PoseStamped = PoseStamped()

            # Convert timestamp to ROS Time
            sec: int = int(timestamp)
            nanosec: int = int((timestamp - sec) * 1e9)
            msg.header.stamp = Time(sec=sec, nanosec=nanosec)
            msg.header.frame_id = "map"

            msg.pose.position.x = float(row["pos_x"])
            msg.pose.position.y = float(row["pos_y"])
            msg.pose.position.z = float(row["pos_z"])

            msg.pose.orientation.x = float(row["orient_x"])
            msg.pose.orientation.y = float(row["orient_y"])
            msg.pose.orientation.z = float(row["orient_z"])
            msg.pose.orientation.w = float(row["orient_w"])

            self.publisher_.publish(msg)
            self.index += 1

        except Exception as e:
            self.get_logger().error(f"Failed to publish pose at index {self.index}: {e}")
            self.index += 1  # Skip problematic line


def main(args=None) -> None:
    """
    Entry point for the ROS 2 node.

    Args:
        args (list | None): Optional list of command-line arguments.
    """
    rclpy.init(args=args)
    node = PosePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
