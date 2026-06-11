"""Dataset-agnostic topological mapper.

Consumes synchronized (image, pose, covariance) samples — wherever they come
from (a real robot's odometry or a dataset player) — and incrementally builds
a :class:`TopoGraph`.

Differences from the thesis implementation, all in service of generality:

- No filename parsing, no polynomial floorplan warping, no re-reading images
  from a dataset directory: the mapper only sees what arrives on its inputs.
- Revisit detection: fixed metric thresholds (``distance_threshold``,
  ``hard_threshold``) are replaced by a chi-square gate under the accumulated
  odometry covariance, combined with a mutual-nearest-neighbor visual check.
- Node descriptors: the running mean of the descriptors of the frames
  assigned to the node, instead of re-extracting features from SIFT-stitched
  panoramas (whose frequent degeneracies polluted both matching and the
  semantic embeddings).
- Rewiring tolerance: derived from the median edge length of the local loop
  instead of a fixed 1.0 m.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gtsam
import numpy as np
from gtsam import BetweenFactorPose2, Pose2, noiseModel

from vts_core.matching import mahalanobis_gate, mutual_nearest_neighbors
from vts_core.motion import normalize_angle
from vts_core.node_detection import AdaptiveValleyDetector, ConnectivityMonitor
from vts_core.topo_graph import Pose2D, TopoGraph, TopoNode


@dataclass
class FrameRecord:
    """A buffered frame: descriptor, pose, covariance snapshot and image."""

    descriptor: np.ndarray
    pose: Pose2D
    covariance: np.ndarray
    image: np.ndarray


class TopologicalMapper:
    """Incremental topological mapper for a single run."""

    def __init__(
        self,
        window_size: int = 30,
        frame_buffer_size: int = 85,
        frame_id: str = "run",
    ) -> None:
        """
        Args:
            window_size: Sliding window size for the connectivity monitor.
            frame_buffer_size: How many recent frames are kept for node
                creation (must exceed window_size).
            frame_id: Identifier of this run's odometry frame.
        """
        self.graph: TopoGraph = TopoGraph(frame_id=frame_id)
        self.current_node_id: int | None = None

        self._monitor: ConnectivityMonitor = ConnectivityMonitor(window_size)
        self._detector: AdaptiveValleyDetector = AdaptiveValleyDetector()
        self._frames: deque[FrameRecord] = deque(maxlen=frame_buffer_size)
        self._frame_count: int = 0
        self._frames_in_node: list[FrameRecord] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process_frame(
        self,
        image_bgr: np.ndarray,
        descriptor: np.ndarray,
        pose: Pose2D,
        covariance: np.ndarray,
    ) -> int | None:
        """Process one synchronized sample.

        Args:
            image_bgr: Camera image.
            descriptor: L2-normalized place descriptor of the image.
            pose: Odometry pose (x, y, theta) at capture time.
            covariance: 3x3 odometry covariance at capture time.

        Returns:
            The id of a node created/updated by this frame, or None.
        """
        record: FrameRecord = FrameRecord(
            descriptor=descriptor.astype(np.float32),
            pose=pose,
            covariance=covariance.astype(np.float64),
            image=image_bgr,
        )
        self._frames.append(record)
        self._frames_in_node.append(record)
        self._frame_count += 1

        lambda_2: float | None = self._monitor.update(record.descriptor)
        if lambda_2 is None:
            return None

        valley_index: int | None = self._detector.step(lambda_2)
        if valley_index is None:
            if len(self.graph.nodes) > 1:
                self._continuous_loop_check(record)
            return None

        return self._create_or_merge_node()

    # ------------------------------------------------------------------ #
    # Node creation / revisit
    # ------------------------------------------------------------------ #
    def _representative_frame(self) -> FrameRecord:
        """Frame with the highest affinity degree inside the current window."""
        offset: int = len(self._frames) - self._monitor.window_length
        index: int = max(0, offset + self._monitor.max_degree_index)
        index = min(index, len(self._frames) - 1)
        return self._frames[index]

    def _create_or_merge_node(self) -> int:
        representative: FrameRecord = self._representative_frame()
        self._monitor.reset_representative()

        # Robust node descriptor: mean of frames assigned since last node.
        if self._frames_in_node:
            stacked: np.ndarray = np.stack(
                [f.descriptor for f in self._frames_in_node]
            )
            mean_descriptor: np.ndarray = stacked.mean(axis=0)
        else:
            mean_descriptor = representative.descriptor
        mean_descriptor = mean_descriptor / max(
            float(np.linalg.norm(mean_descriptor)), 1e-12
        )

        candidate: TopoNode = TopoNode(
            node_id=self.graph.new_node_id(),
            pose=representative.pose,
            visual_features=mean_descriptor,
            pose_covariance=representative.covariance[:2, :2].copy(),
        )
        candidate.add_view(representative.image)
        if self._frames_in_node:
            candidate.add_view(self._frames_in_node[0].image)
            candidate.add_view(self._frames_in_node[-1].image)
        self._frames_in_node = []

        if self.current_node_id is None:
            self.graph.add_node(candidate)
            self.current_node_id = candidate.node_id
            return candidate.node_id

        revisited_id: int | None = self._find_revisit(candidate)
        if revisited_id is not None:
            self._merge_into(candidate, revisited_id)
            self._close_loop(revisited_id)
            self.current_node_id = revisited_id
            return revisited_id

        self.graph.add_node(candidate)
        self.graph.add_edge(self.current_node_id, candidate.node_id)
        self.current_node_id = candidate.node_id
        return candidate.node_id

    def _find_revisit(self, candidate: TopoNode) -> int | None:
        """Probabilistic + visual revisit test against all existing nodes.

        A node is a revisit if it passes the chi-square gate on position
        under the combined covariance AND is a mutual visual nearest
        neighbor of the candidate.
        """
        ids, features = self.graph.feature_matrix()
        matches: list[tuple[int, int, float]] = mutual_nearest_neighbors(
            candidate.visual_features[None, :], features
        )
        if not matches:
            return None

        _, match_index, _ = matches[0]
        matched_id: int = ids[match_index]
        if matched_id == self.current_node_id:
            return None

        matched: TopoNode = self.graph.nodes[matched_id]
        delta: np.ndarray = np.array(candidate.pose[:2]) - np.array(
            matched.pose[:2]
        )
        combined_cov: np.ndarray = (
            candidate.pose_covariance + matched.pose_covariance
        )
        if not mahalanobis_gate(delta, combined_cov):
            return None
        return matched_id

    def _merge_into(self, candidate: TopoNode, target_id: int) -> None:
        """Fuse the candidate into an existing node (running mean of features)."""
        target: TopoNode = self.graph.nodes[target_id]
        fused: np.ndarray = target.visual_features + candidate.visual_features
        target.visual_features = fused / max(float(np.linalg.norm(fused)), 1e-12)
        for view in candidate.views:
            target.add_view(view)
        # Revisit observation reduces positional uncertainty: keep the
        # smaller covariance (trace-wise) of the two.
        if float(np.trace(candidate.pose_covariance)) < float(
            np.trace(target.pose_covariance)
        ):
            target.pose_covariance = candidate.pose_covariance.copy()
        if self.current_node_id is not None and self.current_node_id != target_id:
            self.graph.add_edge(self.current_node_id, target_id)

    def _continuous_loop_check(self, record: FrameRecord) -> None:
        """Between valleys, check whether the robot drifted onto an old node."""
        ids, features = self.graph.feature_matrix()
        matches: list[tuple[int, int, float]] = mutual_nearest_neighbors(
            record.descriptor[None, :], features
        )
        if not matches:
            return
        matched_id: int = ids[matches[0][1]]
        if matched_id == self.current_node_id:
            return
        matched: TopoNode = self.graph.nodes[matched_id]
        delta: np.ndarray = np.array(record.pose[:2]) - np.array(matched.pose[:2])
        combined_cov: np.ndarray = (
            record.covariance[:2, :2] + matched.pose_covariance
        )
        if not mahalanobis_gate(delta, combined_cov):
            return
        if self.current_node_id is not None:
            self.graph.add_edge(self.current_node_id, matched_id)
        self.current_node_id = matched_id
        self._close_loop(matched_id)

    # ------------------------------------------------------------------ #
    # Loop closure optimization & rewiring
    # ------------------------------------------------------------------ #
    def _loop_node_ids(self, closing_id: int) -> list[int]:
        """BFS from current node to closing node — the loop's node set."""
        if self.current_node_id is None:
            return []
        frontier: deque[int] = deque([self.current_node_id])
        parents: dict[int, int | None] = {self.current_node_id: None}
        while frontier:
            node_id: int = frontier.popleft()
            if node_id == closing_id:
                break
            for neighbor in self.graph.adjacency[node_id]:
                if neighbor not in parents:
                    parents[neighbor] = node_id
                    frontier.append(neighbor)
        if closing_id not in parents:
            return []
        path: list[int] = []
        cursor: int | None = closing_id
        while cursor is not None:
            path.append(cursor)
            cursor = parents[cursor]
        return path

    def _close_loop(self, closing_id: int) -> None:
        loop_ids: list[int] = self._loop_node_ids(closing_id)
        if len(loop_ids) < 3:
            return
        loop_set: set[int] = set(loop_ids)
        loop_edges: list[tuple[int, int]] = [
            edge
            for edge in self.graph.edges()
            if edge[0] in loop_set and edge[1] in loop_set
        ]
        self._optimize_poses(loop_ids, loop_edges)
        self._rewire(loop_ids, loop_edges)

    def _optimize_poses(
        self, node_ids: list[int], edges: list[tuple[int, int]]
    ) -> None:
        if len(node_ids) < 3 or len(edges) < 2:
            return

        factor_graph: gtsam.NonlinearFactorGraph = gtsam.NonlinearFactorGraph()
        estimates: gtsam.Values = gtsam.Values()

        anchor: TopoNode = self.graph.nodes[node_ids[0]]
        prior_noise = noiseModel.Diagonal.Sigmas(np.array([1e-6, 1e-6, 1e-6]))
        factor_graph.add(
            gtsam.PriorFactorPose2(anchor.node_id, Pose2(*anchor.pose), prior_noise)
        )

        for id_a, id_b in edges:
            pose_a: Pose2 = Pose2(*self.graph.nodes[id_a].pose)
            pose_b: Pose2 = Pose2(*self.graph.nodes[id_b].pose)
            # Edge noise from the nodes' accumulated covariances rather than
            # a fixed sigma: uncertain segments deform more under closure.
            sigma_xy: float = float(
                np.sqrt(
                    max(
                        np.trace(self.graph.nodes[id_a].pose_covariance)
                        + np.trace(self.graph.nodes[id_b].pose_covariance),
                        1e-4,
                    )
                    / 2.0
                )
            )
            model = noiseModel.Diagonal.Sigmas(
                np.array([sigma_xy, sigma_xy, max(0.05, sigma_xy)])
            )
            factor_graph.add(
                BetweenFactorPose2(id_a, id_b, pose_a.between(pose_b), model)
            )
            if not estimates.exists(id_a):
                estimates.insert(id_a, pose_a)
            if not estimates.exists(id_b):
                estimates.insert(id_b, pose_b)
        if not estimates.exists(anchor.node_id):
            estimates.insert(anchor.node_id, Pose2(*anchor.pose))

        optimizer: gtsam.LevenbergMarquardtOptimizer = (
            gtsam.LevenbergMarquardtOptimizer(
                factor_graph, estimates, gtsam.LevenbergMarquardtParams()
            )
        )
        result: gtsam.Values = optimizer.optimize()

        for node_id in node_ids:
            if result.exists(node_id):
                optimized: Pose2 = result.atPose2(node_id)
                self.graph.nodes[node_id].pose = (
                    float(optimized.x()),
                    float(optimized.y()),
                    normalize_angle(float(optimized.theta())),
                )

    def _rewire(
        self, node_ids: list[int], edges: list[tuple[int, int]]
    ) -> None:
        """Project loop nodes onto loop edges; tolerance = data-driven.

        The rewiring tolerance is half the median length of the loop's edges
        — i.e., a node is attached to an edge only if it lies clearly within
        the edge's own spatial scale, removing the fixed 1.0 m constant.
        """
        if len(node_ids) < 4 or not edges:
            return

        lengths: list[float] = []
        for id_a, id_b in edges:
            pa: np.ndarray = np.array(self.graph.nodes[id_a].pose[:2])
            pb: np.ndarray = np.array(self.graph.nodes[id_b].pose[:2])
            lengths.append(float(np.linalg.norm(pa - pb)))
        tolerance: float = 0.5 * float(np.median(lengths))
        if tolerance <= 1e-9:
            return

        for node_id in node_ids:
            node: TopoNode = self.graph.nodes[node_id]
            point: np.ndarray = np.array(node.pose[:2])
            for id_a, id_b in list(edges):
                if node_id in (id_a, id_b):
                    continue
                pa = np.array(self.graph.nodes[id_a].pose[:2])
                pb = np.array(self.graph.nodes[id_b].pose[:2])
                segment: np.ndarray = pb - pa
                seg_len_sq: float = float(segment @ segment)
                if seg_len_sq < 1e-12:
                    continue
                t: float = float((point - pa) @ segment / seg_len_sq)
                if not 0.0 < t < 1.0:
                    continue
                projection: np.ndarray = pa + t * segment
                if float(np.linalg.norm(point - projection)) >= tolerance:
                    continue
                self.graph.add_edge(id_a, node_id)
                self.graph.add_edge(node_id, id_b)
                self.graph.remove_edge(id_a, id_b)
