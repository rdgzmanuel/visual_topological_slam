"""
Map alignment module for fusing topological graphs.

This module provides the MapAligner class which handles the alignment and fusion
of two topological graphs, including image stitching and feature extraction.
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING

import cv2
import numpy as np
import rclpy
import torch
from scipy.spatial import KDTree
from scipy.spatial.distance import cosine
from torchvision import transforms
from vts_camera.camera import Camera
from vts_graph_building.graph_builder import (
    concat_images,
    process_stitched_image,
    world_to_pixel,
)
from vts_graph_building.node import GraphNodeClass

from vts_map_alignment.graph_class import Graph

if TYPE_CHECKING:
    from rclpy.impl.rcutils_logger import RcutilsLogger


class MapAligner:
    """
    Aligns and fuses two topological graphs into a unified graph.

    This class provides methods for finding matching nodes between graphs,
    fusing similar nodes, and inserting new nodes while maintaining graph
    connectivity.

    Attributes:
        updated_graph: The resulting fused graph after alignment.
    """

    # Default configuration constants
    DEFAULT_POSE_WEIGHT: float = 0.8
    DEFAULT_DISTANCE_THRESHOLD: float = 4.0
    DEFAULT_MIN_MATCHES: int = 4
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.9
    DEFAULT_IMAGE_SIZE: tuple[int, int] = (224, 224)

    def __init__(
        self,
        model_name: str,
        trajectory: str,
        world_limits: tuple[float, float, float, float],
        origin: tuple[int, int],
        map_name: str,
    ) -> None:
        """
        Initialize the MapAligner.

        Args:
            model_name: Name of the model to use for feature extraction.
            trajectory: Trajectory identifier for output naming.
            world_limits: World coordinate limits as (min_x, max_x, min_y, max_y).
            origin: Pixel origin coordinates as (x, y).
            map_name: Name of the map file.
        """
        self._trajectory = trajectory
        self._world_limits = world_limits
        self._origin = origin
        self._map_name = map_name

        self._pose_weight = self.DEFAULT_POSE_WEIGHT
        self._threshold = self.DEFAULT_DISTANCE_THRESHOLD
        self._min_matches = self.DEFAULT_MIN_MATCHES
        self._similarity_threshold = self.DEFAULT_SIMILARITY_THRESHOLD

        self._camera = Camera(model_name)
        self._logger: RcutilsLogger = rclpy.logging.get_logger("MapAlignment")

        self._transform = transforms.Compose([
            transforms.Resize(self.DEFAULT_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.updated_graph: Graph | None = None

    def align_graphs(self, graph_1: Graph, graph_2: Graph) -> None:
        """
        Fuse two topological graphs into one unified graph.

        The first graph is used as the base, and nodes from the second graph
        are integrated into it.

        Args:
            graph_1: The base graph to copy and expand.
            graph_2: The secondary graph whose nodes will be integrated.

        Raises:
            ValueError: If either graph is empty.
        """
        if not graph_1.nodes:
            raise ValueError("Base graph (graph_1) cannot be empty")
        if not graph_2.nodes:
            raise ValueError("Secondary graph (graph_2) cannot be empty")

        self.updated_graph = copy.deepcopy(graph_1)
        self.updated_graph.node_id = max(self.updated_graph.nodes.keys()) + 1
        self._integrate_graph(graph_2)

    def _integrate_graph(self, lookup_graph: Graph) -> None:
        """
        Integrate nodes from a secondary graph into the updated graph.

        Args:
            lookup_graph: The graph whose nodes are being evaluated and integrated.
        """
        if self.updated_graph is None:
            raise RuntimeError("updated_graph must be initialized before integration")

        for node in lookup_graph.nodes.values():
            best_match, score = self._search_best_match(node)

            if best_match is None:
                self._logger.warn(f"No match found for node {node.id}, skipping")
                continue

            if score < self._threshold:
                self._fusion_nodes(node, best_match)
                self.updated_graph.current_node = best_match
            else:
                new_node = copy.deepcopy(node)
                new_node.id = self.updated_graph.node_id

                self._insert_node(new_node, best_match)

                self.updated_graph.nodes[new_node.id] = new_node
                self.updated_graph.current_node = new_node
                self.updated_graph.node_id += 1

    def _insert_node(
        self, new_node: GraphNodeClass, best_match: GraphNodeClass
    ) -> None:
        """
        Insert a new node into the graph with proper edge connections.

        Connects the new node to the best match and attempts to reroute
        existing neighbor edges through this node if appropriate.

        Args:
            new_node: The new node to insert.
            best_match: The existing node to connect to.
        """
        if self.updated_graph is None:
            raise RuntimeError("updated_graph must be initialized")

        next_node = self._find_next_node(best_match, new_node)
        self.updated_graph.edges.append((best_match.id, new_node.id))

        if next_node is None:
            return

        self.updated_graph.edges.append((new_node.id, next_node.id))
        self._remove_edge(best_match.id, next_node.id)
        self._check_next_node_neighbors(next_node, new_node)

    def _remove_edge(self, node_id_1: int, node_id_2: int) -> None:
        """
        Remove edge between two nodes in both directions.

        Args:
            node_id_1: First node ID.
            node_id_2: Second node ID.
        """
        if self.updated_graph is None:
            return

        direct_edge = (node_id_1, node_id_2)
        reverse_edge = (node_id_2, node_id_1)

        if direct_edge in self.updated_graph.edges:
            self.updated_graph.edges.remove(direct_edge)
        if reverse_edge in self.updated_graph.edges:
            self.updated_graph.edges.remove(reverse_edge)

    def _find_next_node(
        self, previous_match: GraphNodeClass, new_node: GraphNodeClass
    ) -> GraphNodeClass | None:
        """
        Find the neighbor node that best aligns with the direction to new_node.

        Args:
            previous_match: The current node in the graph.
            new_node: The new node being inserted.

        Returns:
            The neighbor node to connect through, or None if no suitable match.
        """
        if self.updated_graph is None:
            return None

        direction_vector = np.array(new_node.pose[:2]) - np.array(
            previous_match.pose[:2]
        )
        direction_norm = np.linalg.norm(direction_vector)

        if direction_norm == 0:
            return None

        direction_vector = direction_vector / direction_norm

        neighbors = self._get_node_neighbors(previous_match)
        best_match: GraphNodeClass | None = None
        max_similarity = float("-inf")

        for neighbor in neighbors:
            edge_vector = np.array(neighbor.pose[:2]) - np.array(
                previous_match.pose[:2]
            )
            edge_norm = np.linalg.norm(edge_vector)

            if edge_norm == 0:
                continue

            edge_vector = edge_vector / edge_norm
            similarity = float(np.dot(direction_vector, edge_vector))

            if similarity > self._similarity_threshold and similarity > max_similarity:
                max_similarity = similarity
                best_match = neighbor

        return best_match

    def _get_node_neighbors(self, node: GraphNodeClass) -> list[GraphNodeClass]:
        """
        Get all neighbors of a node from the updated graph.

        Args:
            node: The node to find neighbors for.

        Returns:
            List of neighboring nodes.
        """
        if self.updated_graph is None:
            return []

        neighbors: list[GraphNodeClass] = []

        for edge in self.updated_graph.edges:
            if node.id not in edge:
                continue

            neighbor_id = edge[1] if edge[0] == node.id else edge[0]
            neighbor = self.updated_graph.nodes.get(neighbor_id)

            if neighbor is not None:
                neighbors.append(neighbor)

        return neighbors

    def _search_best_match(
        self, node: GraphNodeClass, k: int = 3
    ) -> tuple[GraphNodeClass | None, float]:
        """
        Find the best matching node using spatial proximity and visual similarity.

        Uses a KD-tree for efficient spatial search and cosine distance for
        visual feature comparison.

        Args:
            node: The query node.
            k: Number of nearest spatial neighbors to consider.

        Returns:
            Tuple of (best matching node, combined similarity score).
            Returns (None, inf) if no match is found.
        """
        if self.updated_graph is None or not self.updated_graph.nodes:
            return None, float("inf")

        if not 0.0 <= self._pose_weight <= 1.0:
            raise ValueError("Pose weight must be between 0 and 1")

        node_list = list(self.updated_graph.nodes.values())
        node_positions = np.array([n.pose[:2] for n in node_list])

        kd_tree = KDTree(node_positions)
        k_actual = min(k, len(node_list))
        distances, indices = kd_tree.query(node.pose[:2], k=k_actual)

        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)

        best_node: GraphNodeClass | None = None
        best_score = float("inf")

        for i, idx in enumerate(indices):
            candidate = node_list[idx]
            pose_distance = float(distances[i])
            visual_distance = float(
                cosine(node.visual_features, candidate.visual_features)
            )

            score = (
                self._pose_weight * pose_distance
                + (1 - self._pose_weight) * visual_distance
            )

            if score < best_score:
                best_score = score
                best_node = candidate

        return best_node, best_score

    def _check_next_node_neighbors(
        self, next_node: GraphNodeClass, new_node: GraphNodeClass
    ) -> None:
        """
        Check if new_node should be inserted between next_node and its neighbors.

        If new_node lies in a similar direction and forms a shorter path to a
        neighbor, it replaces the direct edge with edges through new_node.

        Args:
            next_node: An existing node in the graph.
            new_node: A candidate node to insert between next_node and its neighbors.
        """
        if self.updated_graph is None:
            return

        relevant_edges = [
            e
            for e in self.updated_graph.edges
            if next_node.id in e and new_node.id not in e
        ]

        for edge in relevant_edges:
            neighbor_id = edge[1] if edge[0] == next_node.id else edge[0]
            neighbor = self.updated_graph.nodes.get(neighbor_id)

            if neighbor is None:
                continue

            edge_vector = np.array(neighbor.pose[:2]) - np.array(next_node.pose[:2])
            direction_vector = np.array(neighbor.pose[:2]) - np.array(new_node.pose[:2])

            similarity = float(np.dot(direction_vector, edge_vector))

            edge_distance = self._compute_distance(
                neighbor.pose[0], neighbor.pose[1], next_node.pose[0], next_node.pose[1]
            )
            new_distance = self._compute_distance(
                neighbor.pose[0], neighbor.pose[1], new_node.pose[0], new_node.pose[1]
            )

            if similarity > self._similarity_threshold and edge_distance > new_distance:
                self.updated_graph.edges.append((new_node.id, neighbor.id))
                self._remove_edge(neighbor.id, next_node.id)

    def _fusion_nodes(self, node: GraphNodeClass, best_match: GraphNodeClass) -> None:
        """
        Fuse two nodes by averaging poses and stitching images.

        Updates best_match in place with averaged pose, stitched image,
        and recomputed visual features.

        Args:
            node: Node from the secondary graph.
            best_match: Node from the base graph to update.
        """
        new_pose = self._average_pose(node.pose, best_match.pose)
        new_image = self.stitch_images(node.image, best_match.image)
        tensor_image = process_stitched_image(new_image, self._transform)
        new_visual_features = self._extract_features(tensor_image)

        best_match.pose = new_pose
        best_match.image = new_image
        best_match.visual_features = new_visual_features
        best_match.update_semantics()

    def _extract_features(self, image: torch.Tensor) -> np.ndarray:
        """
        Extract visual features from an image tensor.

        Args:
            image: Input image as a PyTorch tensor.

        Returns:
            Feature vector as a float32 numpy array.
        """
        features = self._camera.extract_features(image)
        features_flat = features.view(-1)

        if features_flat.is_cuda:
            features_list = features_flat.cpu().tolist()
        else:
            features_list = features_flat.tolist()

        return np.array(features_list, dtype=np.float32)

    def generate_map(self) -> None:
        """
        Generate and save a visualization of the aligned graph on a map image.

        Draws node positions as red circles and edges as green lines.

        Raises:
            RuntimeError: If updated_graph is not initialized.
            FileNotFoundError: If the map image file does not exist.
        """
        if self.updated_graph is None:
            raise RuntimeError("Cannot generate map: updated_graph is not initialized")

        map_folder = os.path.join("images/maps", self._map_name)
        output_dir = f"images/final_aligned_maps/{self._map_name[:-4]}"
        output_path = os.path.join(output_dir, f"{self._trajectory}.png")

        if not os.path.exists(map_folder):
            raise FileNotFoundError(f"Map image not found: {map_folder}")

        os.makedirs(output_dir, exist_ok=True)

        map_img = cv2.imread(map_folder)
        if map_img is None:
            raise ValueError(f"Failed to load map image: {map_folder}")

        self._draw_nodes(map_img)
        self._draw_edges(map_img)

        cv2.imwrite(output_path, map_img)

    def _draw_nodes(self, map_img: np.ndarray) -> None:
        """
        Draw nodes on the map image as red circles.

        Args:
            map_img: The map image to draw on (modified in place).
        """
        if self.updated_graph is None:
            return

        for node in self.updated_graph.nodes.values():
            y, x, _ = node.pose  # x, y, _ for saarbruecken_a
            px, py = world_to_pixel(
                -x, y, map_img.shape, self._world_limits, origin=self._origin
            )  # x, y, _ for saarbruecken_a
            cv2.circle(map_img, (px, py), 5, (0, 0, 255), -1)

    def _draw_edges(self, map_img: np.ndarray) -> None:
        """
        Draw edges on the map image as green lines.

        Args:
            map_img: The map image to draw on (modified in place).
        """
        if self.updated_graph is None:
            return

        for idx_1, idx_2 in self.updated_graph.edges:
            node_1 = self.updated_graph.nodes.get(idx_1)
            node_2 = self.updated_graph.nodes.get(idx_2)

            if node_1 is None or node_2 is None:
                continue

            y1, x1, _ = node_1.pose  # x1, y1, _ for saarbruecken_a
            y2, x2, _ = node_2.pose  # x2, y2, _ for saarbruecken_a

            p1 = world_to_pixel(
                -x1, y1, map_img.shape, self._world_limits, origin=self._origin
            )  # x1, y1
            p2 = world_to_pixel(
                -x2, y2, map_img.shape, self._world_limits, origin=self._origin
            )  # x2, y2
            cv2.line(map_img, p1, p2, (0, 255, 0), 2)

    @staticmethod
    def _average_pose(
        pose_1: tuple[float, float, float], pose_2: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """
        Compute the element-wise average of two poses.

        Args:
            pose_1: First pose as (y, x, theta).
            pose_2: Second pose as (y, x, theta).

        Returns:
            Averaged pose as (y, x, theta).
        """
        return (
            (pose_1[0] + pose_2[0]) / 2,
            (pose_1[1] + pose_2[1]) / 2,
            (pose_1[2] + pose_2[2]) / 2,
        )

    @staticmethod
    def _compute_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Compute Euclidean distance between two 2D points.

        Args:
            x1: X coordinate of first point.
            y1: Y coordinate of first point.
            x2: X coordinate of second point.
            y2: Y coordinate of second point.

        Returns:
            Euclidean distance between the points.
        """
        return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    def stitch_images(
        self,
        image_1: np.ndarray,
        image_2: np.ndarray,
        min_matches: int | None = None,
    ) -> np.ndarray:
        """
        Stitch two images using SIFT feature matching and homography.

        Args:
            image_1: First input image (BGR format).
            image_2: Second input image (BGR format).
            min_matches: Minimum good matches required. Defaults to instance setting.

        Returns:
            Stitched image if successful, otherwise a fallback image.
        """
        if min_matches is None:
            min_matches = self._min_matches

        if image_1.size == 0 or image_2.size == 0:
            self._logger.warn("Empty image provided for stitching")
            return image_1 if image_1.size > 0 else image_2

        gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_1, None)
        kp2, des2 = sift.detectAndCompute(gray_2, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return image_1

        good_matches = self._find_good_matches(des1, des2)

        if len(good_matches) < min_matches:
            return concat_images(image_1, image_2)

        homography = self._compute_homography(kp1, kp2, good_matches)

        if homography is None:
            return image_1

        return self._warp_and_blend(image_1, image_2, homography)

    def _find_good_matches(
        self, des1: np.ndarray, des2: np.ndarray
    ) -> list[cv2.DMatch]:
        """
        Find good feature matches using FLANN matcher with ratio test.

        Args:
            des1: Descriptors from first image.
            des2: Descriptors from second image.

        Returns:
            List of good matches that pass the ratio test.
        """
        index_params = {"algorithm": 1, "trees": 5}
        search_params = {"checks": 50}
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except cv2.error:
            return []

        return [m for m, n in matches if m.distance < 0.7 * n.distance]

    def _compute_homography(
        self,
        kp1: list[cv2.KeyPoint],
        kp2: list[cv2.KeyPoint],
        matches: list[cv2.DMatch],
    ) -> np.ndarray | None:
        """
        Compute homography matrix from matched keypoints.

        Args:
            kp1: Keypoints from first image.
            kp2: Keypoints from second image.
            matches: List of matches between keypoints.

        Returns:
            3x3 homography matrix, or None if computation fails.
        """
        if len(matches) < 4:
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None or H.shape != (3, 3):
            return None

        return H

    def _warp_and_blend(
        self, image_1: np.ndarray, image_2: np.ndarray, homography: np.ndarray
    ) -> np.ndarray:
        """
        Warp and blend two images using a homography.

        Args:
            image_1: First image to warp.
            image_2: Second image (reference).
            homography: 3x3 transformation matrix.

        Returns:
            Blended result image, or image_1 if warping produces poor results.
        """
        height, width = image_2.shape[:2]
        warped = cv2.warpPerspective(image_1, homography, (width * 2, height))
        warped[0:height, 0:width] = image_2

        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)

        nonzero_ratio = np.count_nonzero(mask) / mask.size

        if nonzero_ratio < 0.2:
            return image_1

        return self._crop_black_borders(warped, mask)

    @staticmethod
    def _crop_black_borders(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Crop black borders from an image using a binary mask.

        Args:
            image: Image with potential black borders.
            mask: Binary mask indicating valid (non-black) regions.

        Returns:
            Cropped image without black borders.
        """
        coords = cv2.findNonZero(mask)
        if coords is None:
            return image

        x, y, w, h = cv2.boundingRect(coords)
        return image[y : y + h, x : x + w]
