"""Graph building module for visual topological SLAM."""

from __future__ import annotations

import math
import os
from collections import deque
from typing import TYPE_CHECKING

import cv2
import gtsam
import numpy as np
import rclpy.logging
import scipy.sparse
import torch
from gtsam import BetweenFactorPose2, Pose2, noiseModel
from PIL import Image
from scipy.sparse.linalg import eigsh
from torchvision import transforms
from vts_camera.camera import Camera

from vts_graph_building.node import GraphNodeClass

if TYPE_CHECKING:
    from collections.abc import Callable

# Type aliases for clarity
Pose2D = tuple[float, float, float]
Pose2DXY = tuple[float, float]
WorldLimits = tuple[float, float, float, float]
PixelCoord = tuple[int, int]
Edge = tuple[GraphNodeClass, GraphNodeClass]


class GraphBuilder:
    """Builds and maintains a topological graph for visual SLAM."""

    # Class constants
    _SPARSE_MATRIX_THRESHOLD: int = 50
    _DEFAULT_FLANN_TREES: int = 5
    _DEFAULT_FLANN_CHECKS: int = 50
    _LOWE_RATIO_THRESHOLD: float = 0.7
    _RANSAC_THRESHOLD: float = 5.0
    _MIN_NONZERO_RATIO: float = 0.2

    def __init__(
        self,
        n: int,
        gamma_proportion: float,
        delta_proportion: float,
        distance_threshold: float,
        initial_pose: Pose2D,
        world_limits: WorldLimits,
        map_name: str,
        origin: PixelCoord,
        weights: tuple[float, ...],
        trajectory: str,
        model_name: str,
        ext_rewiring: bool,
    ) -> None:
        """
        Initialize the GraphBuilder.

        Args:
            n: Window size for affinity matrix.
            gamma_proportion: Proportion of lambda_2_max for gamma threshold.
            delta_proportion: Proportion of gamma for delta threshold.
            distance_threshold: Distance threshold for node matching.
            initial_pose: Initial pose (x, y, theta).
            world_limits: World coordinate limits (x_min, x_max, y_min, y_max).
            map_name: Name of the map file.
            origin: Pixel coordinates of the origin.
            weights: Polynomial weights for coordinate transformation.
            trajectory: Trajectory name.
            model_name: Model name for feature extraction.
            ext_rewiring: Whether to enable external rewiring.
        """
        # Pose tracking
        self._initial_pose: Pose2D = initial_pose
        self.current_pose: Pose2D = (0.0, 0.0, 0.0)
        self.current_node: GraphNodeClass | None = None
        self._current_image: np.ndarray | None = None
        self.steps: int = 0

        # Graph structure
        self.graph: list[Edge] = []
        self._node_id: int = 0

        # Window and matrices
        self.window_full: bool = False
        self.window_images: np.ndarray | None = None
        self.affinity: np.ndarray | None = None
        self.degree: np.ndarray | None = None
        self._laplacian_sym: np.ndarray | None = None

        # Eigenvalue tracking
        self.eigenvalues: list[float] = []
        self._representative_candidates: list[tuple[int, float]] = []
        self._current_alg_connectivity: float = 0.0

        # Configuration
        self._world_limits: WorldLimits = world_limits
        self._map_name: str = map_name
        self._origin: PixelCoord = origin
        self._weights: tuple[float, ...] = weights
        self._trajectory: str = trajectory

        # Paths
        self._images_path: str = os.path.join(
            "../../project/seq_data/", self._trajectory, "std_cam"
        )

        # Feature detection
        self._sift: cv2.SIFT = cv2.SIFT_create()

        # Window parameters
        self._n: int = n
        self._lambda_2_max: float = n / (n - 1)
        self._gamma: float = gamma_proportion * self._lambda_2_max
        self._delta: float = delta_proportion * self._gamma
        self._distance_threshold: float = distance_threshold

        # Similarity tracking
        self._max_similarity: float = 0.0
        self._max_index: int = 0

        # Peak and valley detection
        self.look_for_maximum: bool = True
        self.max_value: float = float("-inf")
        self.min_value: float = float("inf")
        self.min_idx: int = 0
        self._current_image_idx: int = -1

        # Image buffer
        max_size: int = 85
        self._images_pose: deque[tuple[np.ndarray, Pose2D, np.ndarray]] = deque(
            maxlen=max_size
        )

        # Image stitching parameters
        self._min_matches: int = 4
        self._min_descriptors: int = 2
        self._camera: Camera = Camera(model_name)
        self._image_shape: tuple[int, int, int] | None = None

        # Rewiring parameters
        self._ext_rewiring: bool = ext_rewiring
        self._rewiring_threshold: float = 1.0
        self._external_rewiring_threshold: float = 1.0
        self._hard_threshold: float = 0.5
        self._min_rewire_nodes: int = 3

        # Logger
        self._logger = rclpy.logging.get_logger("GraphBuilder")

        # Image transforms
        self.transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def _apply_polynomial_transform(
        self, value: float, weights: tuple[float, ...]
    ) -> float:
        """
        Apply 5th degree polynomial transformation.

        Args:
            value: Input value.
            weights: Polynomial coefficients (a5, a4, a3, a2, a1, a0).

        Returns:
            Transformed value.
        """
        return (
            weights[0] * (value**5)
            + weights[1] * (value**4)
            + weights[2] * (value**3)
            + weights[3] * (value**2)
            + weights[4] * value
            + weights[5]
        )

    def _transform_coordinates(self, x: float, y: float) -> tuple[float, float]:
        """
        Transform raw coordinates using polynomial weights.

        Args:
            x: Raw x coordinate.
            y: Raw y coordinate.

        Returns:
            Transformed (x, y) coordinates.
        """
        x_transformed: float = self._apply_polynomial_transform(x, self._weights[:6])
        y_transformed: float = self._apply_polynomial_transform(y, self._weights[6:12])
        return x_transformed, y_transformed

    def update_pose(self, v: float, w: float, time_difference: float) -> None:
        """
        Update current pose using odometry measurements.

        Args:
            v: Linear velocity.
            w: Angular velocity.
            time_difference: Time delta between measurements.
        """
        if self.steps == 0:
            prev_x, prev_y, prev_theta = self._initial_pose
        else:
            prev_x, prev_y, prev_theta = self.current_pose

        half_angular_displacement: float = (w * time_difference) / 2
        x: float = (
            prev_x
            + v * np.cos(prev_theta + half_angular_displacement) * time_difference
        )
        y: float = (
            prev_y
            + v * np.sin(prev_theta + half_angular_displacement) * time_difference
        )
        theta: float = self._normalize_angle(prev_theta + w * time_difference)

        self.current_pose = (x, y, theta)
        self.steps += 1
        self._plot_node_on_map(self.current_pose, is_node=False)

    def new_update_pose(self, image_name: str) -> None:
        """
        Update pose by parsing image filename and applying coordinate transform.

        Args:
            image_name: Image filename containing encoded pose information.
        """
        self._current_image = self._convert_image(image_name)

        splitted_msg: list[str] = image_name.split("_")
        raw_x: float = float(splitted_msg[1][1:])
        raw_y: float = float(splitted_msg[2][1:])
        x, y = self._transform_coordinates(raw_x, raw_y)
        theta: float = self._normalize_angle(float(splitted_msg[3][1:5]))

        self.current_pose = (x, y, theta)
        self._plot_node_on_map(self.current_pose, is_node=False)
        self.steps += 1

    def update_pose_odom(self, image_name: str, new_pose: Pose2D) -> None:
        """
        Update pose using odometry data and image.

        Args:
            image_name: Image filename.
            new_pose: New pose from odometry (x, y, theta).
        """
        self._current_image = self._convert_image(image_name)

        raw_x, raw_y, theta = new_pose
        x, y = self._transform_coordinates(raw_x, raw_y)
        theta = self._normalize_angle(theta)

        self.current_pose = (x, y, theta)
        self._plot_node_on_map(self.current_pose, is_node=False)
        self.steps += 1

    def _convert_image(self, image_name: str) -> np.ndarray:
        """
        Load and resize image from filename.

        Args:
            image_name: Name of the image file.

        Returns:
            Loaded image as numpy array.
        """
        image_path: str = os.path.join(self._images_path, image_name)
        image: np.ndarray = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if self._current_image is None and image is not None:
            self._image_shape = image.shape

        if self._image_shape is not None and image is not None:
            image = cv2.resize(image, (self._image_shape[1], self._image_shape[0]))

        return image

    @staticmethod
    def _compute_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Compute Euclidean distance between two 2D points.

        Args:
            x1: First point x coordinate.
            y1: First point y coordinate.
            x2: Second point x coordinate.
            y2: Second point y coordinate.

        Returns:
            Euclidean distance.
        """
        return math.hypot(x1 - x2, y1 - y2)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """
        Normalize angle to [-π, π] range.

        Args:
            angle: Input angle in radians.

        Returns:
            Normalized angle.
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _build_affinity_matrix(self) -> None:
        """Expand affinity matrix by one row and column."""
        if self.affinity is None:
            self.affinity = np.zeros((1, 1), dtype=np.float32)
        else:
            self.affinity = np.pad(
                self.affinity, ((0, 1), (0, 1)), mode="constant", constant_values=0
            )

    def _move_window(self) -> None:
        """Shift window by removing first element and clearing last row/column."""
        self.affinity[:-1, :-1] = self.affinity[1:, 1:]
        self.affinity[-1, :] = 0
        self.affinity[:, -1] = 0
        self.window_images = self.window_images[1:]

    def _compute_similarities(self, array_data: np.ndarray) -> None:
        """
        Update affinity matrix with similarities to new image.

        Args:
            array_data: Normalized feature vector of new image.
        """
        similarities: np.ndarray = np.dot(self.window_images, array_data)
        self.affinity[:-1, -1] = similarities[:-1]
        self.affinity[-1, :-1] = similarities[:-1]
        self.affinity[-1, -1] = 0.0

    def _update_degree_matrix(self) -> None:
        """Update degree matrix and track maximum similarity."""
        self.degree = np.diag(self.affinity.sum(axis=1))
        degree_diag: np.ndarray = np.diag(self.degree)
        max_idx: int = int(np.argmax(degree_diag))

        if degree_diag[max_idx] > self._max_similarity:
            self._max_similarity = degree_diag[max_idx]
            self._max_index = max_idx

    def _update_laplacian_sym_matrix(self) -> None:
        """Compute symmetric normalized Laplacian matrix."""
        degree_diag: np.ndarray = np.diag(self.degree)
        degree_inv_sqrt: np.ndarray = np.diag(
            1.0 / np.sqrt(np.maximum(degree_diag, 1e-10))
        )
        identity: np.ndarray = np.eye(self.affinity.shape[0])
        self._laplacian_sym = (
            identity - degree_inv_sqrt @ self.affinity @ degree_inv_sqrt
        )

    def update_matrices(self, array_data: np.ndarray) -> None:
        """
        Update all matrices with new image features.

        Args:
            array_data: Feature vector of new image.
        """
        norm: float = np.linalg.norm(array_data)
        if norm < 1e-10:
            return

        norm_array_data: np.ndarray = (array_data / norm).astype(np.float32)
        self._images_pose.append((
            norm_array_data,
            self.current_pose,
            self._current_image,
        ))
        self._current_image_idx += 1

        if self.window_images is None:
            self.window_images = np.empty((0, norm_array_data.shape[0]))
        self.window_images = np.vstack([self.window_images, norm_array_data])

        if len(self.window_images) <= self._n:
            self._build_affinity_matrix()
        else:
            self._move_window()

        self._compute_similarities(norm_array_data)
        self._update_degree_matrix()
        self._update_laplacian_sym_matrix()
        self.window_full = len(self.window_images) == self._n

    def _obtain_eigenvalue(self) -> float:
        """
        Compute second smallest eigenvalue of Laplacian (algebraic connectivity).

        Returns:
            Second smallest eigenvalue.
        """
        if self._laplacian_sym.shape[0] > self._SPARSE_MATRIX_THRESHOLD:
            laplacian_sparse = scipy.sparse.csr_matrix(self._laplacian_sym)
            eigenvalues, _ = eigsh(laplacian_sparse, k=2, which="SM")
            lambda_2: float = float(eigenvalues[1])
        else:
            eigenvalues, _ = np.linalg.eigh(self._laplacian_sym)
            lambda_2 = float(eigenvalues[1])

        self.eigenvalues.append(lambda_2)
        return lambda_2

    def look_for_valley(self) -> tuple[float, int]:
        """
        Apply peak-valley detection algorithm.

        Returns:
            Tuple of (current eigenvalue, valley index if found else 0).
        """
        lambda_2: float = self._obtain_eigenvalue()
        valley_idx: int = 0

        if lambda_2 > self.max_value:
            self.max_value = lambda_2
        elif lambda_2 < self.min_value:
            self.min_value = lambda_2
            self.min_idx = self._current_image_idx

        if self.look_for_maximum:
            if (
                self._current_alg_connectivity < (self.max_value - self._delta)
                and self.max_value >= self._gamma
            ):
                self.look_for_maximum = False
                self.min_value = lambda_2
        else:
            if self._current_alg_connectivity > (self.min_value + self._delta):
                self.look_for_maximum = True
                self.max_value = lambda_2
                valley_idx = self.min_idx

        self._current_alg_connectivity = lambda_2
        return lambda_2, valley_idx

    def update_graph(self) -> None:
        """Create new node from selected representative and update graph."""
        idx: int = len(self._images_pose) - len(self.window_images) + self._max_index
        representative: tuple[np.ndarray, Pose2D, np.ndarray] = self._images_pose[idx]
        self._max_similarity = float("-inf")

        new_node: GraphNodeClass = GraphNodeClass(
            id=self._node_id,
            pose=representative[1],
            visual_features=representative[0],
            image=representative[2],
        )

        if self.current_node is None:
            self._node_id += 1
            new_node.update_semantics()
            self.current_node = new_node
            self._plot_node_on_map(self.current_node.pose)
            return

        closest_neighbor: GraphNodeClass = self._search_closest_neighbor(
            new_node.pose, new_node.visual_features
        )
        distance: float = self._compute_distance(
            closest_neighbor.pose[0],
            closest_neighbor.pose[1],
            new_node.pose[0],
            new_node.pose[1],
        )

        if distance < self._distance_threshold:
            self._handle_revisit(new_node, closest_neighbor)
        else:
            self._handle_new_node(new_node)

    def _handle_revisit(
        self, new_node: GraphNodeClass, closest_neighbor: GraphNodeClass
    ) -> None:
        """
        Handle revisit scenario by fusing nodes and rewiring.

        Args:
            new_node: Newly detected node.
            closest_neighbor: Existing node to fuse with.
        """
        closest_neighbor = self._fusion_nodes(new_node, closest_neighbor)
        loop_nodes: list[GraphNodeClass] = self._obtain_loop_nodes(closest_neighbor)
        relevant_edges: list[Edge] = self._obtain_relevant_edges(loop_nodes)

        self._optimize_loop_poses(loop_nodes, relevant_edges)
        self._rewire_graph(loop_nodes, relevant_edges, self._rewiring_threshold)
        self.current_node = closest_neighbor

    def _handle_new_node(self, new_node: GraphNodeClass) -> None:
        """
        Handle new node scenario by adding to graph.

        Args:
            new_node: New node to add.
        """
        self._node_id += 1
        new_node.update_semantics()
        self.graph.append((self.current_node, new_node))

        if self._ext_rewiring:
            relevant_edges: list[Edge] = [
                edge for edge in self.graph if self.current_node in edge
            ]
            self._rewire_graph(
                [new_node], relevant_edges, self._external_rewiring_threshold
            )

            relevant_nodes: list[GraphNodeClass] = list(self.current_node.neighbors)
            self._rewire_graph(
                relevant_nodes,
                [(self.current_node, new_node)],
                self._external_rewiring_threshold,
            )

        new_node.neighbors.add(self.current_node)
        self.current_node.neighbors.add(new_node)
        self.current_node = new_node
        self._plot_node_on_map(self.current_node.pose)

    def _search_closest_neighbor(
        self,
        pose: Pose2D,
        new_visual_features: np.ndarray | None = None,
    ) -> GraphNodeClass:
        """
        Find the closest node in graph to given pose.

        Args:
            pose: Query pose (x, y, theta).
            new_visual_features: Optional visual features for similarity weighting.

        Returns:
            Closest graph node.
        """
        if not self.graph:
            return self.current_node

        unique_nodes: set[GraphNodeClass] = {
            node for edge in self.graph for node in edge
        }
        nodes: list[GraphNodeClass] = list(unique_nodes)

        positions: np.ndarray = np.array(
            [node.pose[:2] for node in nodes], dtype=np.float32
        )
        target_position: np.ndarray = np.array(pose[:2], dtype=np.float32)
        position_distances: np.ndarray = np.linalg.norm(
            positions - target_position, axis=1
        )

        if new_visual_features is not None:
            angles: np.ndarray = np.array(
                [node.pose[2] for node in nodes], dtype=np.float32
            )
            angle_diff: np.ndarray = np.abs(
                (angles - pose[2] + np.pi) % (2 * np.pi) - np.pi
            )
            visual_weights: np.ndarray = 0.5 * (1 - angle_diff / np.pi)

            visual_features: np.ndarray = np.vstack([
                node.visual_features for node in nodes
            ])
            norm_new: float = np.linalg.norm(new_visual_features)
            norm_existing: np.ndarray = np.linalg.norm(visual_features, axis=1)

            visual_similarities: np.ndarray = 1 - (
                visual_features @ new_visual_features
            ) / (norm_existing * norm_new + 1e-8)

            overall_similarities: np.ndarray = (
                1 - visual_weights
            ) * position_distances + visual_weights * visual_similarities
        else:
            overall_similarities = position_distances

        return nodes[int(np.argmin(overall_similarities))]

    def _fusion_nodes(
        self, new_node: GraphNodeClass, closest_neighbor: GraphNodeClass
    ) -> GraphNodeClass:
        """
        Fuse new node with closest neighbor by stitching images.

        Args:
            new_node: New node to fuse.
            closest_neighbor: Existing node to update.

        Returns:
            Updated closest neighbor.
        """
        new_image: np.ndarray = self.stitch_images(
            closest_neighbor.image, new_node.image, min_matches=self._min_matches
        )
        new_image = cv2.resize(new_image, (self._image_shape[1], self._image_shape[0]))
        tensor_image: torch.Tensor = process_stitched_image(new_image, self.transform)
        new_visual_features: np.ndarray = self._extract_features(tensor_image)

        closest_neighbor.image = new_image
        closest_neighbor.visual_features = new_visual_features
        self._plot_node_on_map(closest_neighbor.pose)
        closest_neighbor.update_semantics()

        if self.current_node != closest_neighbor:
            closest_neighbor.neighbors.add(self.current_node)
            self.current_node.neighbors.add(closest_neighbor)
            new_edge: Edge = (self.current_node, closest_neighbor)
            if new_edge not in self.graph:
                self.graph.append(new_edge)

        return closest_neighbor

    def _average_pose(self, pose_1: Pose2D, pose_2: Pose2D) -> Pose2D:
        """
        Compute average of two poses.

        Args:
            pose_1: First pose.
            pose_2: Second pose.

        Returns:
            Averaged pose with normalized angle.
        """
        avg_x: float = (pose_1[0] + pose_2[0]) / 2
        avg_y: float = (pose_1[1] + pose_2[1]) / 2
        avg_theta: float = self._normalize_angle((pose_1[2] + pose_2[2]) / 2)
        return (avg_x, avg_y, avg_theta)

    def _extract_features(self, image: torch.Tensor) -> np.ndarray:
        """
        Extract features from image tensor using camera model.

        Args:
            image: Image tensor.

        Returns:
            Feature vector as numpy array.
        """
        features: torch.Tensor = self._camera.extract_features(image)

        if features.is_cuda:
            feature_list: list[float] = features.view(-1).cpu().tolist()
        else:
            feature_list = features.view(-1).tolist()

        return np.array(feature_list, dtype=np.float32)

    def _obtain_loop_nodes(self, closest_node: GraphNodeClass) -> list[GraphNodeClass]:
        """
        Get all nodes in loop between current node and closest node.

        Args:
            closest_node: Target node for loop closure.

        Returns:
            List of nodes in the loop.
        """
        loop_nodes: set[GraphNodeClass] = set()
        node: GraphNodeClass = self.current_node
        limit: int = len(self.graph)

        if node != closest_node:
            i: int = 1
            while node != closest_node and i <= limit:
                loop_nodes.add(node)
                node = self.graph[-i][0]
                i += 1
            loop_nodes.add(closest_node)

        return list(loop_nodes)

    def _obtain_relevant_edges(self, loop_nodes: list[GraphNodeClass]) -> list[Edge]:
        """
        Get edges where both endpoints are in loop nodes.

        Args:
            loop_nodes: List of nodes in the loop.

        Returns:
            List of relevant edges.
        """
        loop_node_set: set[GraphNodeClass] = set(loop_nodes)
        return [
            edge
            for edge in self.graph
            if edge[0] in loop_node_set and edge[1] in loop_node_set
        ]

    def _rewire_graph(
        self,
        loop_nodes: list[GraphNodeClass],
        relevant_edges: list[Edge],
        threshold: float,
    ) -> None:
        """
        Rewire graph by projecting nodes onto edges.

        Args:
            loop_nodes: Nodes to potentially rewire.
            relevant_edges: Edges to consider for rewiring.
            threshold: Distance threshold for rewiring.
        """
        if len(loop_nodes) <= self._min_rewire_nodes:
            return

        for node in loop_nodes:
            x, y, _ = node.pose
            projections: dict[Edge, Pose2DXY] = self._get_projections(
                node, relevant_edges
            )

            for edge, projection in projections.items():
                if projection is None:
                    continue

                distance: float = self._compute_distance(
                    x, y, projection[0], projection[1]
                )

                if distance < threshold:
                    self._apply_rewiring(node, edge, projection)

    def _apply_rewiring(
        self, node: GraphNodeClass, edge: Edge, projection: Pose2DXY
    ) -> None:
        """
        Apply rewiring operation for a single node-edge pair.

        Args:
            node: Node being rewired.
            edge: Edge to split.
            projection: Projection point on edge.
        """
        self.graph.append((edge[0], node))
        self.graph.append((node, edge[1]))
        edge[0].neighbors.add(node)
        edge[1].neighbors.add(node)
        node.neighbors.add(edge[0])
        node.neighbors.add(edge[1])

        self._aggregate_node(node, projection)

        if edge in self.graph:
            self.graph.remove(edge)
            reverse_edge: Edge = (edge[1], edge[0])
            if reverse_edge in self.graph:
                self.graph.remove(reverse_edge)
                if edge[1] in edge[0].neighbors:
                    edge[0].neighbors.remove(edge[1])
                if edge[0] in edge[1].neighbors:
                    edge[1].neighbors.remove(edge[0])

    def _get_projections(
        self,
        node: GraphNodeClass,
        relevant_edges: list[Edge],
    ) -> dict[Edge, Pose2DXY]:
        """
        Compute valid projections of node onto edges.

        Args:
            node: Node to project.
            relevant_edges: Edges to project onto.

        Returns:
            Dictionary mapping edges to projection coordinates.
        """
        projections: dict[Edge, Pose2DXY] = {}

        for edge in relevant_edges:
            n1, n2 = edge

            if node in (n1, n2):
                continue

            p1: np.ndarray = np.array(n1.pose[:2])
            p2: np.ndarray = np.array(n2.pose[:2])
            edge_vector: np.ndarray = p2 - p1
            edge_length_sq: float = float(np.dot(edge_vector, edge_vector))

            if edge_length_sq < 1e-10:
                continue

            projection: Pose2DXY = self._project_node(node, edge)
            proj: np.ndarray = np.array(projection)
            proj_vector: np.ndarray = proj - p1

            t: float = float(np.dot(proj_vector, edge_vector) / edge_length_sq)

            if 0.0 <= t <= 1.0:
                projections[edge] = (float(proj[0]), float(proj[1]))

        return projections

    def _project_node(self, node: GraphNodeClass, edge: Edge) -> Pose2DXY:
        """
        Project node onto edge line.

        Args:
            node: Node to project.
            edge: Edge defining the line.

        Returns:
            Projection coordinates (x, y).
        """
        x1, y1, _ = edge[0].pose
        x2, y2, _ = edge[1].pose
        x, y, _ = node.pose

        v_x: float = x2 - x1
        v_y: float = y2 - y1
        u_x: float = x - x1
        u_y: float = y - y1

        dot_product: float = u_x * v_x + u_y * v_y
        v_squared_magnitude: float = v_x**2 + v_y**2
        projection_scalar: float = dot_product / (v_squared_magnitude + 1e-6)

        proj_x: float = x1 + projection_scalar * v_x
        proj_y: float = y1 + projection_scalar * v_y

        return (proj_x, proj_y)

    def _optimize_loop_poses(
        self,
        nodes: list[GraphNodeClass],
        graph_edges: list[Edge],
    ) -> None:
        """
        Apply pose graph optimization for loop closure.

        Args:
            nodes: Nodes in the loop.
            graph_edges: Edges between nodes.
        """
        if len(nodes) < 3 or len(graph_edges) < 2:
            return

        graph: gtsam.NonlinearFactorGraph = gtsam.NonlinearFactorGraph()
        initial_estimates: gtsam.Values = gtsam.Values()

        if not nodes:
            return

        fixed_node: GraphNodeClass = nodes[0]
        fixed_pose: Pose2 = Pose2(*fixed_node.pose)
        prior_noise = noiseModel.Diagonal.Sigmas([1e-6, 1e-6, 1e-6])
        graph.add(gtsam.PriorFactorPose2(fixed_node.id, fixed_pose, prior_noise))
        initial_estimates.insert(fixed_node.id, fixed_pose)

        for node1, node2 in graph_edges:
            pose1: Pose2 = Pose2(*node1.pose)
            pose2: Pose2 = Pose2(*node2.pose)
            relative_pose: Pose2 = pose1.between(pose2)
            model = noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1])
            graph.add(BetweenFactorPose2(node1.id, node2.id, relative_pose, model))

            if not initial_estimates.exists(node1.id):
                initial_estimates.insert(node1.id, pose1)
            if not initial_estimates.exists(node2.id):
                initial_estimates.insert(node2.id, pose2)

        params: gtsam.LevenbergMarquardtParams = gtsam.LevenbergMarquardtParams()
        optimizer: gtsam.LevenbergMarquardtOptimizer = (
            gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
        )
        result: gtsam.Values = optimizer.optimize()

        for node in nodes:
            if result.exists(node.id):
                optimized_pose: Pose2 = result.atPose2(node.id)
                node.pose = (
                    optimized_pose.x(),
                    optimized_pose.y(),
                    optimized_pose.theta(),
                )

    def _aggregate_node(self, node: GraphNodeClass, projection: Pose2DXY) -> None:
        """
        Aggregate node with projection by averaging pose and stitching images.

        Args:
            node: Node to aggregate.
            projection: Projection coordinates.
        """
        new_projection_pose: Pose2D = (
            projection[0],
            projection[1],
            self._normalize_angle(node.pose[2] + np.pi),
        )
        new_pose: Pose2D = self._average_pose(node.pose, new_projection_pose)
        node.pose = new_pose

        image: np.ndarray = self._find_closest_image(projection)
        new_image: np.ndarray = self.stitch_images(
            node.image, image, min_matches=self._min_matches
        )
        new_image = cv2.resize(new_image, (self._image_shape[1], self._image_shape[0]))
        tensor_image: torch.Tensor = process_stitched_image(new_image, self.transform)
        new_visual_features: np.ndarray = self._extract_features(tensor_image)

        node.image = new_image
        node.visual_features = new_visual_features
        node.update_semantics()

    def _find_closest_image(self, query_coords: Pose2DXY) -> np.ndarray:
        """
        Find image with closest position to query coordinates.

        Args:
            query_coords: Target (x, y) coordinates.

        Returns:
            Image at closest position.
        """
        query: np.ndarray = np.array(query_coords)
        positions: np.ndarray = np.array([pose[1][:2] for pose in self._images_pose])
        deltas: np.ndarray = positions - query
        squared_distances: np.ndarray = np.einsum("ij,ij->i", deltas, deltas)
        min_idx: int = int(np.argmin(squared_distances))
        return self._images_pose[min_idx][2]

    def check_pose(self) -> None:
        """Check if current pose matches an existing node for loop closure."""
        x, y, theta = self.current_pose
        match: GraphNodeClass | None = self._search_closest_neighbor((x, y, theta))

        if match is None or match == self.current_node:
            return

        distance: float = self._compute_distance(x, y, match.pose[0], match.pose[1])

        if distance >= self._hard_threshold:
            return

        loop_nodes: list[GraphNodeClass] = self._obtain_loop_nodes(match)
        relevant_edges: list[Edge] = self._obtain_relevant_edges(loop_nodes)
        self._rewire_graph(loop_nodes, relevant_edges, self._rewiring_threshold)

        new_edge: Edge = (self.current_node, match)
        match.neighbors.add(self.current_node)
        self.current_node.neighbors.add(match)

        if new_edge not in self.graph:
            self.graph.append(new_edge)

        self.current_node = match

        new_image: np.ndarray = self.stitch_images(
            self.current_node.image,
            self._current_image,
            min_matches=self._min_matches,
        )
        new_image = cv2.resize(new_image, (self._image_shape[1], self._image_shape[0]))
        tensor_image: torch.Tensor = process_stitched_image(new_image, self.transform)
        new_visual_features: np.ndarray = self._extract_features(tensor_image)

        self.current_node.image = new_image
        self.current_node.visual_features = new_visual_features
        self.current_node.update_semantics()

    def _plot_node_on_map(self, pose: Pose2D, is_node: bool = True) -> None:
        """
        Plot node or trajectory point on map image.

        Args:
            pose: Pose to plot.
            is_node: If True, plot as node (larger, red); else as trajectory (blue).
        """
        map_folder: str = os.path.join("images/maps", self._map_name)
        output_path: str = os.path.join(
            f"images/running_maps/{self._map_name[:-4]}",
            f"{self._trajectory}_nodes.png",
        )

        if self.steps != 0:
            map_folder = output_path

        map_img: np.ndarray = cv2.imread(map_folder)
        if map_img is None:
            return

        x, y, _ = pose
        px, py = world_to_pixel(x, y, map_img.shape, self._world_limits, self._origin)

        if is_node:
            cv2.circle(map_img, (px, py), 5, (0, 0, 255), -1)
        else:
            cv2.circle(map_img, (px, py), 1, (255, 0, 0), -1)

        cv2.imwrite(output_path, map_img)

    def generate_map(self) -> None:
        """Generate final path image with trajectory and nodes."""
        map_folder: str = os.path.join("images/maps", self._map_name)
        file_name: str = f"final_{self._trajectory}.png"
        output_path: str = os.path.join(
            f"images/adjusted_maps/{self._map_name[:-4]}", file_name
        )

        map_img: np.ndarray = cv2.imread(map_folder)
        if map_img is None:
            return

        for _, pose, _ in self._images_pose:
            x, y, _ = pose
            px, py = world_to_pixel(
                x, y, map_img.shape, self._world_limits, self._origin
            )
            cv2.circle(map_img, (px, py), 1, (255, 0, 0), -1)

        for node, _ in self.graph:
            x, y, _ = node.pose
            px, py = world_to_pixel(
                x, y, map_img.shape, self._world_limits, self._origin
            )
            cv2.circle(map_img, (px, py), 5, (0, 0, 255), -1)

        cv2.imwrite(output_path, map_img)

    def generate_map_edges(self) -> None:
        """Generate final map image with nodes and edges."""
        map_folder: str = os.path.join("images/maps", self._map_name)
        file_name: str = f"final_{self._trajectory}.png"
        output_path: str = os.path.join("images/final_edges_maps", file_name)

        map_img: np.ndarray = cv2.imread(map_folder)
        if map_img is None:
            return

        for node, _ in self.graph:
            x, y, _ = node.pose
            px, py = world_to_pixel(
                x, y, map_img.shape, self._world_limits, self._origin
            )
            cv2.circle(map_img, (px, py), 5, (0, 0, 255), -1)

        for node_1, node_2 in self.graph:
            if node_1 is not None and node_2 is not None:
                x1, y1, _ = node_1.pose
                x2, y2, _ = node_2.pose
                p1: PixelCoord = world_to_pixel(
                    x1, y1, map_img.shape, self._world_limits, self._origin
                )
                p2: PixelCoord = world_to_pixel(
                    x2, y2, map_img.shape, self._world_limits, self._origin
                )
                cv2.line(map_img, p1, p2, (0, 255, 0), 2)

        cv2.imwrite(output_path, map_img)

    def stitch_images(
        self, image_1: np.ndarray, image_2: np.ndarray, min_matches: int = 4
    ) -> np.ndarray:
        """
        Stitch two images using SIFT feature matching and homography.

        Args:
            image_1: First image (BGR format).
            image_2: Second image (BGR format).
            min_matches: Minimum good matches required.

        Returns:
            Stitched image or fallback.
        """
        if image_1.size == 0 or image_2.size == 0:
            return image_1

        gray_1: np.ndarray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        gray_2: np.ndarray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self._sift.detectAndCompute(gray_1, None)
        kp2, des2 = self._sift.detectAndCompute(gray_2, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return image_1

        index_params: dict[str, int] = {
            "algorithm": 1,
            "trees": self._DEFAULT_FLANN_TREES,
        }
        search_params: dict[str, int] = {"checks": self._DEFAULT_FLANN_CHECKS}
        flann: cv2.FlannBasedMatcher = cv2.FlannBasedMatcher(
            index_params, search_params
        )

        try:
            matches: list[list[cv2.DMatch]] = flann.knnMatch(des1, des2, k=2)
        except cv2.error:
            return image_1

        good_matches: list[cv2.DMatch] = [
            m
            for m, n in matches
            if m.distance < self._LOWE_RATIO_THRESHOLD * n.distance
        ]

        if len(good_matches) < min_matches:
            return concat_images(image_1, image_2)

        src_pts: np.ndarray = np.float32([
            kp1[m.queryIdx].pt for m in good_matches
        ]).reshape(-1, 1, 2)
        dst_pts: np.ndarray = np.float32([
            kp2[m.trainIdx].pt for m in good_matches
        ]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self._RANSAC_THRESHOLD)

        if H is None or H.shape != (3, 3):
            return image_1

        height, width = image_2.shape[:2]
        warped_img1: np.ndarray = cv2.warpPerspective(image_1, H, (width * 2, height))
        warped_img1[0:height, 0:width] = image_2

        gray_warped: np.ndarray = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY)
        _, mask_thresh = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)
        nonzero_ratio: float = np.count_nonzero(mask_thresh) / mask_thresh.size

        if nonzero_ratio < self._MIN_NONZERO_RATIO:
            return image_1

        return crop_black_borders(warped_img1, mask_thresh)


def crop_black_borders(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Crop black borders from image using mask.

    Args:
        image: Image with potential black borders.
        mask: Binary mask indicating valid regions.

    Returns:
        Cropped image.
    """
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y : y + h, x : x + w]


def concat_images(
    image_1: np.ndarray, image_2: np.ndarray, axis: int = 1
) -> np.ndarray:
    """
    Concatenate two images horizontally or vertically.

    Args:
        image_1: First image.
        image_2: Second image.
        axis: 0 for vertical, 1 for horizontal (default).

    Returns:
        Concatenated image.
    """
    if axis == 1 and image_1.shape[0] != image_2.shape[0]:
        h: int = min(image_1.shape[0], image_2.shape[0])
        image_1 = cv2.resize(image_1, (int(image_1.shape[1] * h / image_1.shape[0]), h))
        image_2 = cv2.resize(image_2, (int(image_2.shape[1] * h / image_2.shape[0]), h))
    elif axis == 0 and image_1.shape[1] != image_2.shape[1]:
        w: int = min(image_1.shape[1], image_2.shape[1])
        image_1 = cv2.resize(image_1, (w, int(image_1.shape[0] * w / image_1.shape[1])))
        image_2 = cv2.resize(image_2, (w, int(image_2.shape[0] * w / image_2.shape[1])))

    return np.concatenate((image_1, image_2), axis=axis)


def process_stitched_image(
    image: np.ndarray,
    transform: Callable[[Image.Image], torch.Tensor],
) -> torch.Tensor:
    """
    Convert OpenCV image to PyTorch tensor with transforms.

    Args:
        image: BGR image array.
        transform: Torchvision transform pipeline.

    Returns:
        Transformed image tensor.
    """
    image_rgb: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil: Image.Image = Image.fromarray(image_rgb)
    return transform(image_pil)


def world_to_pixel(
    x: float,
    y: float,
    map_shape: tuple[int, ...],
    world_limits: WorldLimits,
    origin: PixelCoord,
) -> PixelCoord:
    """
    Convert world coordinates to pixel coordinates.

    Args:
        x: World x coordinate.
        y: World y coordinate.
        map_shape: Shape of map image.
        world_limits: World coordinate limits (x_min, x_max, y_min, y_max).
        origin: Pixel coordinates of world origin.

    Returns:
        Pixel coordinates (px, py).
    """
    x_min, x_max, y_min, y_max = world_limits
    map_h, map_w = map_shape[:2]
    origin_x, origin_y = origin

    scale_x: float = map_w / (x_max - x_min)
    scale_y: float = map_h / (y_max - y_min)

    px: int = int(origin_x + (x * scale_x))
    py: int = int(origin_y - (y * scale_y))

    return (px, py)
