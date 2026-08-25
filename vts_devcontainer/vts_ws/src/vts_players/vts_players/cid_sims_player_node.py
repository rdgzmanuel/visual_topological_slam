"""Lossless ROS2 player for native CID-SIMS sequence directories.

The adapter publishes the same standard topics as the COLD player. Dataset
knowledge stays here; :mod:`vts_core` and the graph builder remain unchanged.
"""

from __future__ import annotations

import json
import os
from contextlib import suppress

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String

from vts_core.motion import OdometryUncertaintyParams, Pose2D
from vts_players.cid_sims_data import (
    load_ground_truth,
    load_wheel_odometry,
    timestamped_color_images,
)


class CidSimsPlayerNode(Node):
    """Publish one or more CID-SIMS sequences with recorded wheel odometry."""

    def __init__(self) -> None:
        super().__init__("cid_sims_player")
        self.declare_parameter("dataset_root", "")
        self.declare_parameter("sequences", [""])
        self.declare_parameter("color_subdir", "color")
        self.declare_parameter("ground_truth_file", "groundtruth.txt")
        self.declare_parameter("odometry_file", "odom.txt")
        self.declare_parameter("frame_stride", 6)
        self.declare_parameter("max_ground_truth_gap_s", 0.1)
        self.declare_parameter("max_odometry_gap_s", 0.2)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("inter_sequence_pause_s", 2.0)
        self.declare_parameter(
            "odometry_uncertainty", [0.025, 0.005, 0.01, 0.0025]
        )

        self._dataset_root = self.get_parameter("dataset_root").value
        self._sequences = list(self.get_parameter("sequences").value)
        self._color_subdir = self.get_parameter("color_subdir").value
        self._ground_truth_file = self.get_parameter("ground_truth_file").value
        self._odometry_file = self.get_parameter("odometry_file").value
        self._frame_stride = int(self.get_parameter("frame_stride").value)
        self._max_gt_gap_s = float(
            self.get_parameter("max_ground_truth_gap_s").value
        )
        self._max_odom_gap_s = float(self.get_parameter("max_odometry_gap_s").value)
        rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self._pause_ticks = int(
            float(self.get_parameter("inter_sequence_pause_s").value) * rate_hz
        )
        uncertainty = list(self.get_parameter("odometry_uncertainty").value)
        if self._frame_stride < 1:
            raise ValueError("frame_stride must be at least one")
        if len(uncertainty) != 4:
            raise ValueError("odometry_uncertainty must contain four coefficients")
        self._uncertainty_params = OdometryUncertaintyParams(*uncertainty)

        self._bridge = CvBridge()
        self._image_pub = self.create_publisher(Image, "/camera/image", 10)
        self._odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self._gt_pub = self.create_publisher(
            PoseStamped, "/ground_truth_pose", 10
        )
        self._label_pub = self.create_publisher(
            String, "/dataset/room_label", 10
        )
        self._done_pub = self.create_publisher(
            String, "/dataset/sequence_done", 10
        )
        self.create_subscription(
            Empty, "/mapping/frame_processed", self._on_frame_processed, 10
        )

        self._sequence_index = 0
        self._frame_index = 0
        self._pause_remaining = 0
        self._waiting_for_frame_ack = False
        self._consumer_connected = False
        self._frames: list[
            tuple[float, str, Pose2D, Pose2D, np.ndarray]
        ] = []
        self._load_sequence(0)
        self._timer = self.create_timer(1.0 / rate_hz, self._tick)

    def _load_sequence(self, index: int) -> None:
        sequence = self._sequences[index]
        sequence_dir = os.path.join(self._dataset_root, sequence)
        if not os.path.isdir(sequence_dir):
            raise FileNotFoundError(
                f"CID-SIMS sequence directory not found: {sequence_dir}"
            )
        ground_truth = load_ground_truth(
            os.path.join(sequence_dir, self._ground_truth_file)
        )
        odometry = load_wheel_odometry(
            os.path.join(sequence_dir, self._odometry_file),
            self._uncertainty_params,
        )
        images = timestamped_color_images(
            os.path.join(sequence_dir, self._color_subdir)
        )
        selected = images[:: self._frame_stride]
        frames: list[tuple[float, str, Pose2D, Pose2D, np.ndarray]] = []
        skipped = 0
        for timestamp, image_path in selected:
            try:
                gt_pose, _ = ground_truth.at(timestamp, self._max_gt_gap_s)
                odom_pose, covariance = odometry.at(
                    timestamp, self._max_odom_gap_s
                )
            except ValueError:
                skipped += 1
                continue
            if covariance is None:
                raise RuntimeError("wheel odometry stream has no covariance")
            frames.append(
                (timestamp, image_path, gt_pose, odom_pose, covariance)
            )
        if not frames:
            raise RuntimeError(
                f"No synchronized CID-SIMS frames found in {sequence_dir}"
            )
        self._frames = frames
        self._frame_index = 0
        self._consumer_connected = False
        self.get_logger().info(
            f"Sequence {sequence}: {len(images)} RGB frames, "
            f"{len(frames)} selected at stride {self._frame_stride}, "
            f"{skipped} outside synchronization limits"
        )

    def _on_frame_processed(self, _message: Empty) -> None:
        self._waiting_for_frame_ack = False

    def _tick(self) -> None:
        if not self._consumer_connected:
            self._consumer_connected = (
                self._image_pub.get_subscription_count() > 0
                and self._odom_pub.get_subscription_count() > 0
            )
            if not self._consumer_connected:
                return
        if self._waiting_for_frame_ack:
            return
        if self._pause_remaining > 0:
            self._pause_remaining -= 1
            return
        if self._frame_index >= len(self._frames):
            self._announce_sequence_done()
            return

        _timestamp, path, gt_pose, odom_pose, covariance = self._frames[
            self._frame_index
        ]
        self._frame_index += 1
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            status = os.stat(path)
            cloud_hint = (
                " The file occupies no local disk blocks; download it from "
                "cloud storage or mark the dataset as always available offline."
                if getattr(status, "st_blocks", 1) == 0
                else ""
            )
            raise RuntimeError(f"Unreadable CID-SIMS image: {path}.{cloud_hint}")

        stamp = self.get_clock().now().to_msg()
        self._waiting_for_frame_ack = True
        self._label_pub.publish(String(data=""))

        gt_msg = PoseStamped()
        gt_msg.header.stamp = stamp
        gt_msg.header.frame_id = "world"
        gt_msg.pose.position.x = gt_pose[0]
        gt_msg.pose.position.y = gt_pose[1]
        gt_msg.pose.orientation.z = float(np.sin(gt_pose[2] / 2.0))
        gt_msg.pose.orientation.w = float(np.cos(gt_pose[2] / 2.0))
        self._gt_pub.publish(gt_msg)

        image_msg = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = f"cid_sims_seq_{self._sequence_index}"
        self._image_pub.publish(image_msg)

        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = "odom"
        odom_msg.pose.pose.position.x = odom_pose[0]
        odom_msg.pose.pose.position.y = odom_pose[1]
        odom_msg.pose.pose.orientation.z = float(np.sin(odom_pose[2] / 2.0))
        odom_msg.pose.pose.orientation.w = float(np.cos(odom_pose[2] / 2.0))
        flat = np.zeros(36, dtype=np.float64)
        flat[0], flat[1], flat[5] = covariance[0]
        flat[6], flat[7], flat[11] = covariance[1]
        flat[30], flat[31], flat[35] = covariance[2]
        odom_msg.pose.covariance = flat.tolist()
        self._odom_pub.publish(odom_msg)

    def _announce_sequence_done(self) -> None:
        is_last = self._sequence_index >= len(self._sequences) - 1
        payload = {
            "sequence_index": self._sequence_index,
            "sequence_name": self._sequences[self._sequence_index],
            "is_last": is_last,
            "frame_count": len(self._frames),
        }
        self._done_pub.publish(String(data=json.dumps(payload)))
        self.get_logger().info(f"Sequence done: {json.dumps(payload)}")
        if is_last:
            self._timer.cancel()
            return
        self._sequence_index += 1
        self._pause_remaining = self._pause_ticks
        self._load_sequence(self._sequence_index)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CidSimsPlayerNode()
    with suppress(KeyboardInterrupt):
        rclpy.spin(node)
    with suppress(KeyboardInterrupt):
        node.destroy_node()
    with suppress(Exception):
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
