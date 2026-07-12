"""Graph builder ROS2 node (dataset-agnostic).

Subscribes to standard topics — works identically with the COLD player, any
other dataset player, or a live robot:

- ``/camera/image``           sensor_msgs/Image
- ``/odom``                   nav_msgs/Odometry  (with pose covariance)
- ``/ground_truth_pose``      geometry_msgs/PoseStamped (optional, attached
                              to nodes for evaluation only — never used by
                              the mapping algorithm)
- ``/dataset/sequence_done``  std_msgs/String    (JSON; end-of-run signal,
                              replacing the old 20 s image timeout)

For each completed sequence the resulting graph is saved to
``<output_dir>/graph_<i>.pkl`` and a JSON event is published on
``/mapping/graph_ready``. After the last sequence it also publishes
``{"all_done": true}`` so the alignment node knows to start.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import suppress

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from vts_core.features import FeatureExtractor, build_extractor
from vts_core.mapper import TopologicalMapper
from vts_core.metrics import map_footprint
from vts_core.motion import Pose2D


def _yaw_from_quaternion(z: float, w: float) -> float:
    return float(2.0 * np.arctan2(z, w))


class GraphBuilderNode(Node):
    """Builds one topological graph per incoming sequence."""

    def __init__(self) -> None:
        super().__init__("graph_builder")

        self.declare_parameter("window_size", 30)
        self.declare_parameter("valley_k", 1.5)
        self.declare_parameter("merge_radius", 2.0)
        self.declare_parameter("visual_outlier_k", 2.0)
        self.declare_parameter("optimize", False)
        self.declare_parameter("extractor", "dinov2")
        self.declare_parameter("model_name", "")
        self.declare_parameter("encoder_path", "")
        self.declare_parameter("output_dir", "output/graphs")
        self.declare_parameter("run_name", "run")

        self._window_size: int = (
            self.get_parameter("window_size").get_parameter_value().integer_value
        )
        self._valley_k: float = (
            self.get_parameter("valley_k").get_parameter_value().double_value
        )
        self._merge_radius: float = (
            self.get_parameter("merge_radius").get_parameter_value().double_value
        )
        self._visual_outlier_k: float = (
            self.get_parameter("visual_outlier_k").get_parameter_value().double_value
        )
        extractor_spec: str = (
            self.get_parameter("extractor").get_parameter_value().string_value
        )
        model_name: str = (
            self.get_parameter("model_name").get_parameter_value().string_value
        )
        encoder_path: str = (
            self.get_parameter("encoder_path").get_parameter_value().string_value
        )
        self._output_dir: str = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )
        self._run_name: str = (
            self.get_parameter("run_name").get_parameter_value().string_value
        )
        os.makedirs(self._output_dir, exist_ok=True)

        self._extractor: FeatureExtractor = build_extractor(
            extractor_spec, model_name, encoder_path
        )
        self._bridge: CvBridge = CvBridge()

        self._sequence_index: int = 0
        self._processed: int = 0
        self._mapper: TopologicalMapper = self._new_mapper()
        self._latest_gt: Pose2D | None = None
        self._latest_label: str | None = None
        self._node_gt: dict[int, Pose2D] = {}

        image_sub: Subscriber = Subscriber(self, Image, "/camera/image")
        odom_sub: Subscriber = Subscriber(self, Odometry, "/odom")
        self._synchronizer: ApproximateTimeSynchronizer = (
            ApproximateTimeSynchronizer([image_sub, odom_sub], queue_size=30, slop=0.2)
        )
        self._synchronizer.registerCallback(self._on_frame)

        self.create_subscription(
            PoseStamped, "/ground_truth_pose", self._on_ground_truth, 10
        )
        self.create_subscription(
            String, "/dataset/room_label", self._on_room_label, 10
        )
        self.create_subscription(
            String, "/dataset/sequence_done", self._on_sequence_done, 10
        )
        self._ready_pub = self.create_publisher(String, "/mapping/graph_ready", 10)

    # ------------------------------------------------------------------ #
    def _new_mapper(self) -> TopologicalMapper:
        return TopologicalMapper(
            window_size=self._window_size,
            valley_k=self._valley_k,
            merge_radius=self._merge_radius,
            visual_outlier_k=self._visual_outlier_k,
            frame_id=f"{self._run_name}_seq{self._sequence_index}",
        )

    def _on_ground_truth(self, msg: PoseStamped) -> None:
        self._latest_gt = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            _yaw_from_quaternion(
                float(msg.pose.orientation.z), float(msg.pose.orientation.w)
            ),
        )

    def _on_frame(self, image_msg: Image, odom_msg: Odometry) -> None:
        image: np.ndarray = self._bridge.imgmsg_to_cv2(
            image_msg, desired_encoding="bgr8"
        )
        pose: Pose2D = (
            float(odom_msg.pose.pose.position.x),
            float(odom_msg.pose.pose.position.y),
            _yaw_from_quaternion(
                float(odom_msg.pose.pose.orientation.z),
                float(odom_msg.pose.pose.orientation.w),
            ),
        )
        flat: list[float] = list(odom_msg.pose.covariance)
        covariance: np.ndarray = np.array(
            [
                [flat[0], flat[1], 0.0],
                [flat[6], flat[7], 0.0],
                [0.0, 0.0, flat[35]],
            ],
            dtype=np.float64,
        )

        start: float = time.perf_counter()
        descriptor: np.ndarray = self._extractor.extract(image)
        elapsed: float = time.perf_counter() - start
        self._processed += 1
        if self._processed % 25 == 0:
            self.get_logger().info(
                f"{self._processed} frames processed, "
                f"last extraction {elapsed:.3f}s"
            )
        touched_id: int | None = self._mapper.process_frame(
            image, descriptor, pose, covariance, gt_pose=self._latest_gt
        )
        if touched_id is not None:
            node = self._mapper.graph.nodes[touched_id]
            # Evaluation-only ground truth (never used to map). Use the node's
            # own gt_pose: the GT of its representative (medoid) frame, which
            # is the exact frame that fixed the node's pose. Falling back to
            # the latest GT only if the mapper carried none.
            node_gt: Pose2D | None = node.gt_pose or self._latest_gt
            if node_gt is not None:
                self._node_gt.setdefault(touched_id, node_gt)
            if node.room_label is None and self._latest_label is not None:
                node.room_label = self._latest_label

    def _on_room_label(self, msg: String) -> None:
        self._latest_label = msg.data or None

    # ------------------------------------------------------------------ #
    def _on_sequence_done(self, msg: String) -> None:
        initial_error, final_error = self._mapper.finalize()
        self.get_logger().info(
            f"Pose-graph optimization (numpy SE2): error "
            f"{initial_error:.2f} -> {final_error:.2f}"
        )
        eig_path: str = os.path.join(
            self._output_dir, f"graph_{self._sequence_index}_lambda2.npy"
        )
        np.save(eig_path, np.array(self._mapper._monitor.eigenvalues))
        self.get_logger().info(
            f"DIAG: frames_processed={self._mapper._frame_count}, "
            f"nodes={len(self._mapper.graph.nodes)}, "
            f"edges={len(self._mapper.graph.edges())}, "
            f"lambda2 series saved to {eig_path}"
        )
        info: dict[str, object] = json.loads(msg.data)
        graph_path: str = os.path.join(
            self._output_dir, f"graph_{self._sequence_index}.pkl"
        )
        self._mapper.graph.save(graph_path)
        # Single-run pipeline: the per-sequence graph is also the final map the
        # language/evaluation tools consume (no separate alignment step).
        final_path: str = os.path.join(self._output_dir, "final_graph.pkl")
        self._mapper.graph.save(final_path)

        # Computational-cost summary for the PRISM-TopoMap comparison (Table V).
        perf: dict[str, float] = self._mapper.performance_stats()
        footprint: dict[str, float] = map_footprint(self._mapper.graph)
        perf["map_disk_bytes_full"] = float(os.path.getsize(final_path))
        perf["map_disk_bytes_descriptors_only"] = footprint[
            "descriptors_only_pickle_bytes"
        ]
        perf["map_in_memory_bytes"] = footprint["in_memory_bytes"]
        perf_path: str = os.path.join(
            self._output_dir, f"graph_{self._sequence_index}_performance.json"
        )
        with open(perf_path, "w") as f:
            json.dump(perf, f, indent=2)
        self.get_logger().info(
            "PERF: map_update={map_update_time_ms:.2f} ms/frame, "
            "loop_closure_search={loop_closure_time_ms:.2f} ms, "
            "map_size_full={full:.2f} MB, "
            "map_size_descriptors_only={desc:.3f} MB".format(
                **perf,
                full=perf["map_disk_bytes_full"] / 1e6,
                desc=perf["map_disk_bytes_descriptors_only"] / 1e6,
            )
        )
        gt_path: str = os.path.join(
            self._output_dir, f"graph_{self._sequence_index}_node_gt.json"
        )
        with open(gt_path, "w") as f:
            json.dump({str(k): list(v) for k, v in self._node_gt.items()}, f)

        event: dict[str, object] = {
            "graph_path": graph_path,
            "sequence_index": self._sequence_index,
            "all_done": bool(info.get("is_last", False)),
        }
        out: String = String()
        out.data = json.dumps(event)
        self._ready_pub.publish(out)
        self.get_logger().info(f"Graph ready: {event}")

        self._sequence_index += 1
        self._processed = 0
        self._mapper = self._new_mapper()
        self._node_gt = {}


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: GraphBuilderNode = GraphBuilderNode()
    with suppress(KeyboardInterrupt):
        rclpy.spin(node)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
