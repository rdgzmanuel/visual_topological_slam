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
``/mapping/graph_ready``. After the last sequence it publishes
``{"all_done": true}`` and exits cleanly.
"""

from __future__ import annotations

import json
import os
import signal
import time
from contextlib import suppress

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, String

from vts_core.features import FeatureExtractor, build_extractor
from vts_core.mapper import TopologicalMapper
from vts_core.metrics import map_footprint
from vts_core.motion import Pose2D
from vts_core.pose_graph import OptimizationResult


def _yaw_from_quaternion(z: float, w: float) -> float:
    return float(2.0 * np.arctan2(z, w))


class GraphBuilderNode(Node):
    """Builds one topological graph per incoming sequence."""

    def __init__(self) -> None:
        super().__init__("graph_builder")

        self.declare_parameter("window_size", 30)
        self.declare_parameter("valley_k", 2.0)
        self.declare_parameter("visual_outlier_k", 2.0)
        self.declare_parameter("optimize", True)
        self.declare_parameter("optimizer_backend", "gtsam")
        self.declare_parameter("gate_mode", "both")
        self.declare_parameter("naive_threshold", 0.7)
        self.declare_parameter("visual_model", "vmf")
        self.declare_parameter("dino_model", "dinov2_vits14")
        self.declare_parameter("dino_device", "auto")
        self.declare_parameter("feature_backend", "dino_cls")
        self.declare_parameter("dino_layer", -1)
        self.declare_parameter("use_ground_truth", False)
        self.declare_parameter("output_dir", "output/graphs")
        self.declare_parameter("run_name", "run")
        # Exit the process once the last sequence is done (lets scripted runs
        # like run_experiments.sh proceed without a manual Ctrl-C; the launch
        # file shuts the whole pipeline down when this node exits).
        self.declare_parameter("exit_when_done", True)

        self._window_size: int = (
            self.get_parameter("window_size").get_parameter_value().integer_value
        )
        self._valley_k: float = (
            self.get_parameter("valley_k").get_parameter_value().double_value
        )
        self._visual_outlier_k: float = (
            self.get_parameter("visual_outlier_k").get_parameter_value().double_value
        )
        self._optimize: bool = (
            self.get_parameter("optimize").get_parameter_value().bool_value
        )
        self._optimizer_backend: str = (
            self.get_parameter("optimizer_backend")
            .get_parameter_value()
            .string_value
        )
        self._gate_mode: str = (
            self.get_parameter("gate_mode").get_parameter_value().string_value
        )
        self._naive_threshold: float = (
            self.get_parameter("naive_threshold").get_parameter_value().double_value
        )
        self._visual_model = str(self.get_parameter("visual_model").value)
        dino_model: str = (
            self.get_parameter("dino_model").get_parameter_value().string_value
        )
        dino_device: str = (
            self.get_parameter("dino_device").get_parameter_value().string_value
        )
        feature_backend: str = (
            self.get_parameter("feature_backend").get_parameter_value().string_value
        )
        dino_layer: int = (
            self.get_parameter("dino_layer").get_parameter_value().integer_value
        )
        self._feature_backend = feature_backend
        self._dino_model = dino_model
        self._dino_layer = dino_layer
        self._use_ground_truth: bool = (
            self.get_parameter("use_ground_truth").get_parameter_value().bool_value
        )
        self._output_dir: str = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )
        self._run_name: str = (
            self.get_parameter("run_name").get_parameter_value().string_value
        )
        self._exit_when_done: bool = (
            self.get_parameter("exit_when_done").get_parameter_value().bool_value
        )
        self._shutdown_timer = None
        os.makedirs(self._output_dir, exist_ok=True)

        extractor_start: float = time.perf_counter()
        self._extractor: FeatureExtractor = build_extractor(
            dino_model,
            device=dino_device,
            backend=feature_backend,
            layer=dino_layer,
        )
        self._extractor_init_s: float = time.perf_counter() - extractor_start
        self.get_logger().info(
            f"Visual descriptor: {feature_backend}, model={dino_model}, "
            f"layer={dino_layer}, device={self._extractor.device} "
            f"(requested: {dino_device})"
        )
        self._bridge: CvBridge = CvBridge()

        self._sequence_index: int = 0
        self._processed: int = 0
        self._extraction_time_s: float = 0.0
        self._mapper: TopologicalMapper = self._new_mapper()
        self._latest_gt: Pose2D | None = None
        self._latest_label: str | None = None
        self._node_gt: dict[int, Pose2D] = {}

        # Create the acknowledgement publisher before exposing input
        # subscriptions. Once the dataset player observes those subscriptions,
        # this publisher is guaranteed to exist for the first processed frame.
        self._frame_processed_pub = self.create_publisher(
            Empty, "/mapping/frame_processed", 10
        )
        image_sub: Subscriber = Subscriber(self, Image, "/camera/image")
        odom_sub: Subscriber = Subscriber(self, Odometry, "/odom")
        synchronized_subscribers = [image_sub, odom_sub]
        if self._use_ground_truth:
            synchronized_subscribers.append(
                Subscriber(self, PoseStamped, "/ground_truth_pose")
            )
        self._synchronizer: ApproximateTimeSynchronizer = (
            ApproximateTimeSynchronizer(
                synchronized_subscribers, queue_size=30, slop=0.05
            )
        )
        self._synchronizer.registerCallback(self._on_frame)

        if not self._use_ground_truth:
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
        # Offline dataset players may use this acknowledgement for lossless
        # flow control. Live robots can ignore it; mapping input remains the
        # same standard image and odometry topics.

    # ------------------------------------------------------------------ #
    def _new_mapper(self) -> TopologicalMapper:
        return TopologicalMapper(
            window_size=self._window_size,
            valley_k=self._valley_k,
            visual_outlier_k=self._visual_outlier_k,
            optimize=self._optimize,
            optimizer_backend=self._optimizer_backend,
            gate_mode=self._gate_mode,
            naive_threshold=self._naive_threshold,
            visual_model=self._visual_model,
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

    def _on_frame(
        self,
        image_msg: Image,
        odom_msg: Odometry,
        ground_truth_msg: PoseStamped | None = None,
    ) -> None:
        if ground_truth_msg is not None:
            self._on_ground_truth(ground_truth_msg)
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
                [flat[0], flat[1], flat[5]],
                [flat[6], flat[7], flat[11]],
                [flat[30], flat[31], flat[35]],
            ],
            dtype=np.float64,
        )

        start: float = time.perf_counter()
        descriptor: np.ndarray = self._extractor.extract(image)
        elapsed: float = time.perf_counter() - start
        self._extraction_time_s += elapsed
        self._processed += 1
        if self._processed % 25 == 0:
            self.get_logger().info(
                f"{self._processed} frames processed, "
                f"last extraction {elapsed:.3f}s"
            )
        touched_id: int | None = self._mapper.process_frame(
            image,
            descriptor,
            pose,
            covariance,
            gt_pose=self._latest_gt,
            room_label=self._latest_label,
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
        self._frame_processed_pub.publish(Empty())

    def _on_room_label(self, msg: String) -> None:
        self._latest_label = msg.data or None

    # ------------------------------------------------------------------ #
    def _on_sequence_done(self, msg: String) -> None:
        info: dict[str, object] = json.loads(msg.data)
        expected_frames = info.get("frame_count")
        if expected_frames is not None:
            actual_frames = self._mapper._frame_count
            if actual_frames != int(expected_frames):
                raise RuntimeError(
                    "Incomplete frame stream: mapper processed "
                    f"{actual_frames}/{int(expected_frames)} frames"
                )
        # The detector confirms a boundary after the valley itself. Flush the
        # remaining post-valley frames before either graph snapshot is saved.
        flushed_id = self._mapper.finalize_nodes()
        if flushed_id is not None:
            flushed_gt = self._mapper.graph.nodes[flushed_id].gt_pose
            if flushed_gt is not None:
                self._node_gt.setdefault(flushed_id, flushed_gt)
        # Snapshot the map BEFORE optimization so the raw-odometry node
        # placement can be evaluated against the optimized one (RMSE
        # before/after in the paper).
        noopt_path: str = os.path.join(
            self._output_dir, f"graph_{self._sequence_index}_noopt.pkl"
        )
        self._mapper.graph.save(noopt_path)
        optimization: OptimizationResult = self._mapper.optimize_graph()
        self.get_logger().info(
            f"Pose-graph optimization ({optimization.backend} SE2): error "
            f"{optimization.initial_error:.2f} -> "
            f"{optimization.final_error:.2f}; "
            f"iterations={optimization.iterations}, "
            f"converged={optimization.converged}"
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
        graph_path: str = os.path.join(
            self._output_dir, f"graph_{self._sequence_index}.pkl"
        )
        self._mapper.graph.save(graph_path)
        # Single-run pipeline: the per-sequence graph is also the final map the
        # language/evaluation tools consume (no separate alignment step).
        final_path: str = os.path.join(self._output_dir, "final_graph.pkl")
        self._mapper.graph.save(final_path)

        # Full runtime summary, including the frozen DINOv2 forward pass.
        perf: dict[str, object] = self._mapper.performance_stats()
        perf["encoder_time_ms_per_frame"] = (
            1000.0 * self._extraction_time_s / max(self._processed, 1)
        )
        perf["encoder_initialization_s"] = self._extractor_init_s
        perf["feature_backend"] = self._feature_backend
        perf["dino_model"] = self._dino_model
        perf["dino_layer"] = self._dino_layer
        perf["visual_model"] = self._visual_model
        perf["end_to_end_time_ms_per_frame"] = (
            perf["encoder_time_ms_per_frame"] + perf["map_update_time_ms"]
        )
        perf["optimizer_initial_error"] = optimization.initial_error
        perf["optimizer_final_error"] = optimization.final_error
        perf["optimizer_iterations"] = float(optimization.iterations)
        perf["optimizer_converged"] = float(optimization.converged)
        perf["optimizer_backend"] = optimization.backend
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
            "PERF: end_to_end={end_to_end_time_ms_per_frame:.2f} ms/frame, "
            "encoder={encoder_time_ms_per_frame:.2f} ms/frame, "
            "map_update={map_update_time_ms:.2f} ms/frame, "
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
        self._extraction_time_s = 0.0
        self._mapper = self._new_mapper()
        self._node_gt = {}

        if event["all_done"] and self._exit_when_done:
            # Grace period so DDS delivers the graph_ready message, then end
            # the spin. The launch file reacts to this process exiting by
            # shutting down the whole pipeline (player included).
            self.get_logger().info("All sequences done; exiting in 2 s.")
            self._shutdown_timer = self.create_timer(2.0, self._request_shutdown)

    def _request_shutdown(self) -> None:
        # ``rclpy.try_shutdown()`` from inside a timer callback can invalidate
        # the context without waking the executor that is currently spinning;
        # the process then remains alive and launch never receives
        # ``OnProcessExit``. SIGINT follows the already-tested manual Ctrl+C
        # path through ``main`` and therefore guarantees a clean process exit.
        if self._shutdown_timer is not None:
            self._shutdown_timer.cancel()
        self.get_logger().info("Shutdown grace period complete; exiting now.")
        os.kill(os.getpid(), signal.SIGINT)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: GraphBuilderNode = GraphBuilderNode()
    with suppress(KeyboardInterrupt, ExternalShutdownException):
        rclpy.spin(node)
    with suppress(Exception, KeyboardInterrupt):
        node.destroy_node()
    with suppress(Exception):
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
