"""COLD dataset player node.

This is the *only* place in the entire workspace that knows anything about
the COLD dataset. It reads a sequence directory, parses the ground-truth
pose encoded in each filename (``t<sec>.<usec>_x<x>_y<y>_a<angle>.jpeg`` —
note: filenames carry NO room label; COLD distributes labels in separate
annotation files, supported below via the ``labels_file`` parameter), and
publishes:

- ``/camera/image``           sensor_msgs/Image   (BGR8, stamped)
- ``/odom``                   nav_msgs/Odometry   (realistic, drifting)
- ``/ground_truth_pose``      geometry_msgs/PoseStamped (evaluation only)
- ``/dataset/sequence_done``  std_msgs/String     (JSON: sequence index, last?)

The drifting odometry comes from :class:`vts_core.motion.OdometrySimulator`
(probabilistic odometry motion model), replacing the dataset's unusable
odometry files while remaining statistically realistic. Any other dataset —
or a real robot — only needs to provide the same three topics for the rest
of the pipeline to work unchanged.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import suppress

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from vts_core.motion import OdometryNoiseParams, OdometrySimulator, Pose2D

_FILENAME_PATTERN: re.Pattern[str] = re.compile(
    r"t(?P<t>\d+\.\d+)_x(?P<x>-?\d+\.?\d*)_y(?P<y>-?\d+\.?\d*)_a(?P<a>-?\d+\.?\d*)"
)


def parse_cold_filename(name: str) -> tuple[float, Pose2D] | None:
    """Extract (timestamp, ground-truth pose) from a COLD image filename."""
    match: re.Match[str] | None = _FILENAME_PATTERN.search(name)
    if match is None:
        return None
    return (
        float(match.group("t")),
        (float(match.group("x")), float(match.group("y")), float(match.group("a"))),
    )


# Canonical COLD place classes, copied verbatim (including order) from the
# encoder's prepare_data so player labels match the training vocabulary.
COLD_CLASSES: tuple[str, ...] = (
    "CR", "2PO", "RL", "TL", "TR", "LO", "1PO", "KT", "CNR", "PA", "LAB", "ST",
)


def canonical_label(raw: str) -> str:
    """Canonicalize a COLD place label exactly like the encoder's training code.

    First class in ``COLD_CLASSES`` that is a substring of the raw label
    (e.g. ``2PO1-A`` -> ``2PO``, ``CR-A`` -> ``CR``); falls back to stripping
    the building-part suffix and trailing digits for labels outside the list.
    """
    raw = raw.strip()
    for class_name in COLD_CLASSES:
        if class_name in raw:
            return class_name
    base: str = raw.split("-")[0]
    return base.rstrip("0123456789")


def load_room_labels(labels_file: str) -> dict[str, str]:
    """Load a COLD ``localization/places.lst`` file.

    Real format (one line per image): ``<image_filename> <label>``, e.g.
    ``t1152903982.078005_x2.289920_y-0.274201_a0.002522.jpeg CR-A``.

    Returns:
        Mapping image filename (basename, as written in the file) ->
        canonical label. Empty if the file is absent.
    """
    if not labels_file or not os.path.exists(labels_file):
        return {}
    labels: dict[str, str] = {}
    with open(labels_file) as f:
        for line in f:
            parts: list[str] = line.strip().split()
            if len(parts) < 2:
                continue
            labels[os.path.basename(parts[0])] = canonical_label(parts[1])
    return labels


class ColdPlayerNode(Node):
    """Publishes one or more COLD sequences as standard ROS2 topics."""

    def __init__(self) -> None:
        super().__init__("cold_player")

        self.declare_parameter("dataset_root", "")
        self.declare_parameter("sequences", [""])
        self.declare_parameter("images_subdir", "std_cam")
        self.declare_parameter("labels_subpath", "localization/places.lst")
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("inter_sequence_pause_s", 5.0)
        self.declare_parameter("odom_seed", 17)
        self.declare_parameter("alpha", [0.05, 0.01, 0.02, 0.005])

        self._dataset_root: str = (
            self.get_parameter("dataset_root").get_parameter_value().string_value
        )
        self._sequences: list[str] = list(
            self.get_parameter("sequences").get_parameter_value().string_array_value
        )
        self._images_subdir: str = (
            self.get_parameter("images_subdir").get_parameter_value().string_value
        )
        self._labels_subpath: str = (
            self.get_parameter("labels_subpath").get_parameter_value().string_value
        )
        rate_hz: float = (
            self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        )
        self._pause_ticks: int = int(
            self.get_parameter("inter_sequence_pause_s")
            .get_parameter_value()
            .double_value
            * rate_hz
        )
        seed: int = int(
            self.get_parameter("odom_seed").get_parameter_value().integer_value
        )
        alpha: list[float] = list(
            self.get_parameter("alpha").get_parameter_value().double_array_value
        )

        self._noise_params: OdometryNoiseParams = OdometryNoiseParams(
            alpha1=alpha[0], alpha2=alpha[1], alpha3=alpha[2], alpha4=alpha[3]
        )
        self._seed: int = seed

        self._bridge: CvBridge = CvBridge()
        self._image_pub = self.create_publisher(Image, "/camera/image", 10)
        self._odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self._gt_pub = self.create_publisher(PoseStamped, "/ground_truth_pose", 10)
        self._done_pub = self.create_publisher(String, "/dataset/sequence_done", 10)
        self._label_pub = self.create_publisher(String, "/dataset/room_label", 10)

        self._labels: dict[str, str] = {}

        self._sequence_index: int = 0
        self._frame_index: int = 0
        self._pause_remaining: int = 0
        self._frames: list[tuple[float, Pose2D, str]] = []
        self._simulator: OdometrySimulator = OdometrySimulator(
            self._noise_params, seed=self._seed
        )
        self._load_sequence(0)

        self._timer = self.create_timer(1.0 / rate_hz, self._tick)

    # ------------------------------------------------------------------ #
    def _load_sequence(self, index: int) -> None:
        sequence: str = self._sequences[index]
        images_dir: str = os.path.join(
            self._dataset_root, sequence, self._images_subdir
        )
        frames: list[tuple[float, Pose2D, str]] = []
        for name in sorted(os.listdir(images_dir)):
            parsed: tuple[float, Pose2D] | None = parse_cold_filename(name)
            if parsed is None:
                continue
            timestamp, pose = parsed
            frames.append((timestamp, pose, os.path.join(images_dir, name)))
        if not frames:
            raise RuntimeError(f"No parsable COLD images in {images_dir}")
        self._frames = frames
        self._frame_index = 0
        self._labels = load_room_labels(
            os.path.join(self._dataset_root, sequence, self._labels_subpath)
        )
        if not self._labels:
            self.get_logger().warn(
                f"No places.lst found for {sequence}; nodes will carry no "
                "room labels and retrieval metrics will be unavailable."
            )
        # New run, new odometry: each sequence gets an independent simulator
        # (deterministic per-sequence seed for reproducibility).
        self._simulator = OdometrySimulator(
            self._noise_params, seed=self._seed + index
        )
        self.get_logger().info(
            f"Sequence {sequence}: {len(frames)} frames, "
            f"{len(self._labels)} labels loaded."
        )

    def _room_label_of(self, image_path: str) -> str | None:
        return self._labels.get(os.path.basename(image_path))

    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        if self._pause_remaining > 0:
            self._pause_remaining -= 1
            return

        if self._frame_index >= len(self._frames):
            self._announce_sequence_done()
            return

        timestamp, gt_pose, path = self._frames[self._frame_index]
        self._frame_index += 1

        image: np.ndarray | None = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warn(f"Unreadable image skipped: {path}")
            return

        noisy_pose, covariance = self._simulator.step(gt_pose)

        stamp = self.get_clock().now().to_msg()

        image_msg = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = f"cold_seq_{self._sequence_index}"
        self._image_pub.publish(image_msg)

        odom_msg: Odometry = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = "odom"
        odom_msg.pose.pose.position.x = noisy_pose[0]
        odom_msg.pose.pose.position.y = noisy_pose[1]
        odom_msg.pose.pose.orientation.z = float(np.sin(noisy_pose[2] / 2.0))
        odom_msg.pose.pose.orientation.w = float(np.cos(noisy_pose[2] / 2.0))
        flat: np.ndarray = np.zeros(36, dtype=np.float64)
        flat[0] = covariance[0, 0]
        flat[1] = covariance[0, 1]
        flat[6] = covariance[1, 0]
        flat[7] = covariance[1, 1]
        flat[35] = covariance[2, 2]
        odom_msg.pose.covariance = flat.tolist()
        self._odom_pub.publish(odom_msg)

        label: str | None = self._room_label_of(path)
        if label is not None:
            label_msg: String = String()
            label_msg.data = label
            self._label_pub.publish(label_msg)

        gt_msg: PoseStamped = PoseStamped()
        gt_msg.header.stamp = stamp
        gt_msg.header.frame_id = "world"
        gt_msg.pose.position.x = gt_pose[0]
        gt_msg.pose.position.y = gt_pose[1]
        gt_msg.pose.orientation.z = float(np.sin(gt_pose[2] / 2.0))
        gt_msg.pose.orientation.w = float(np.cos(gt_pose[2] / 2.0))
        self._gt_pub.publish(gt_msg)

    def _announce_sequence_done(self) -> None:
        is_last: bool = self._sequence_index >= len(self._sequences) - 1
        payload: str = json.dumps(
            {
                "sequence_index": self._sequence_index,
                "sequence_name": self._sequences[self._sequence_index],
                "is_last": is_last,
            }
        )
        message: String = String()
        message.data = payload
        self._done_pub.publish(message)
        self.get_logger().info(f"Sequence done: {payload}")

        if is_last:
            self._timer.cancel()
            return
        self._sequence_index += 1
        self._pause_remaining = self._pause_ticks
        self._load_sequence(self._sequence_index)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: ColdPlayerNode = ColdPlayerNode()
    with suppress(KeyboardInterrupt):
        rclpy.spin(node)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
