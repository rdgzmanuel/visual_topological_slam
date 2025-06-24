#!/usr/bin/env python3
"""
data_logger_node.py

ROS 2 node that logs camera images and odometry data to disk in a structured format.
"""

import csv
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image


class DataLogger(Node):
    """
    ROS 2 node that stores camera frames and odometry into the topological_map directory.
    """

    def __init__(self) -> None:
        """
        Initialize the DataLogger node, subscriptions, and storage paths.
        """
        super().__init__("data_logger")

        # Directory and file paths
        self.root_dir: Path = Path.home() / "topological_map"
        self.images_dir: Path = self.root_dir / "images"
        self.odom_dir: Path = self.root_dir / "odometry"
        self.odom_file: Path = self.odom_dir / "odometry.csv"

        # ROS topic names and queues
        self.img_topic: str = "/zed/zed_node/rgb/raw/image"
        self.odom_topic: str = "odom"
        self.img_queue: int = 10
        self.odom_queue: int = 10

        self.last_image_save_time: float = 0.0  # in seconds (ROS time)
        self.image_save_interval: float = 0.2   # 200 ms


        # Utility
        self.bridge: CvBridge = CvBridge()

        # File writer
        self.csv_file: Optional[csv._Writer] = None  # type for .close()
        self.csv_writer: Optional[csv._writer] = None

        self._initialise_filesystem()

        self.create_subscription(Image, self.img_topic, self._handle_image, self.img_queue)
        self.create_subscription(Odometry, self.odom_topic, self._handle_odom, self.odom_queue)

        self.csv_file = self.odom_file.open("a", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        if self.csv_file.tell() == 0:
            self.csv_writer.writerow([
                "timestamp",
                "pos_x", "pos_y", "pos_z",
                "orient_x", "orient_y", "orient_z", "orient_w",
                "twist_lin_x", "twist_lin_y", "twist_lin_z",
                "twist_ang_x", "twist_ang_y", "twist_ang_z"
            ])

        self.get_logger().warn("Data-logger node initialized.")

    def _handle_image(self, msg: Image) -> None:
        """
        Callback for storing camera image messages as .png files, at most one every 0.2 seconds.
        Images are resized to 640x360 resolution before saving to reduce storage usage.

        Args:
            msg (Image): The received ROS image message.
        """
        timestamp_float, timestamp_str = self._to_timestamp(msg.header.stamp)

        # Only save if 0.2 seconds have passed since last saved image
        if timestamp_float - self.last_image_save_time < self.image_save_interval:
            return

        self.last_image_save_time = timestamp_float

        img: np.ndarray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Resize to 640x360
        resized_img: np.ndarray = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)

        filename: Path = self.images_dir / f"{timestamp_str}.png"
        success: bool = cv2.imwrite(str(filename), resized_img)

        if not success:
            self.get_logger().error(f"Failed to save resized image: {filename}")



    def _handle_odom(self, msg: Odometry) -> None:
        """
        Callback for logging odometry data into a CSV file.

        Args:
            msg (Odometry): The received ROS odometry message.
        """
        timestamp_float: float
        timestamp_str: str
        timestamp_float, _ = self._to_timestamp(msg.header.stamp)

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        t = msg.twist.twist

        if self.csv_writer:
            self.csv_writer.writerow([
                f"{timestamp_float:.9f}",
                p.x, p.y, p.z,
                q.x, q.y, q.z, q.w,
                t.linear.x, t.linear.y, t.linear.z,
                t.angular.x, t.angular.y, t.angular.z
            ])
            self.csv_file.flush()

    def _to_timestamp(self, stamp: Time) -> tuple[float, str]:
        """
        Convert ROS Time to float seconds and a filename-friendly string.

        Args:
            stamp (Time): ROS timestamp object.

        Returns:
            tuple[float, str]: (float_timestamp, "sec_nsec" formatted string)
        """
        secs: int = stamp.sec
        nsecs: int = stamp.nanosec
        return secs + nsecs * 1e-9, f"{secs}_{nsecs:09d}"

    def _initialise_filesystem(self) -> None:
        """
        Ensure that the topological_map directory structure exists.
        """
        for d in (self.images_dir, self.odom_dir):
            d.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """
        Cleanly close file resources before node shutdown.
        """
        if self.csv_file:
            self.csv_file.close()


def main(args: Optional[list[str]] = None) -> None:
    """
    Entry point for the data logger node.

    Args:
        args (Optional[list[str]]): Command-line arguments passed by ROS 2 launch system.
    """
    rclpy.init(args=args)
    data_logger: DataLogger = DataLogger()

    try:
        rclpy.spin(data_logger)
    except KeyboardInterrupt:
        pass
    finally:
        data_logger.close()
        data_logger.destroy_node()
        rclpy.try_shutdown()
