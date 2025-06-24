import numpy as np
import copy
import os
import cv2
import rclpy
import torch
from scipy.spatial import KDTree
from scipy.spatial.distance import cosine

from vts_graph_building.node import GraphNodeClass
from vts_map_alignment.graph_class import Graph
from vts_graph_building.graph_builder import concat_images ,\
    process_stitched_image, world_to_pixel
from vts_camera.camera import Camera


class MapAligner:
    def __init__(self, model_name: str, trajectory: str,
                 world_limits: tuple[float, float, float, float], origin: tuple[int, int],
                 map_name: str) -> None:
        """
        
        """
        self._trajectory: str = trajectory
        self._world_limits: tuple[float, float, float, float] = world_limits
        self._origin: tuple[int, int] = origin
        self._map_name: str = map_name

        # Neighbor finding
        self._pose_weight: float = 0.8
        self._threshold: float = 4.0

        # Image stitching
        self._min_matches: int = 4
        self._camera: Camera = Camera(model_name)

        # Map alignment
        self._similarity_threshold: float = 0.9

        self._logger = rclpy.logging.get_logger('MapAlignment')


    def align_graphs(self, graph_1: Graph, graph_2: Graph) -> None:
        """
        Fuses two topological graphs into one unified graph stored in self.updated_graph.

        Args:
            graph_1 (Graph): The base graph to copy and expand.
            graph_2 (Graph): The secondary graph whose nodes will be integrated.
        """
        self.updated_graph: Graph = copy.deepcopy(graph_1)
        self.updated_graph.node_id = max(self.updated_graph.nodes.keys()) + 1
        self._update_graph(graph_2)


    def _update_graph(self, lookup_graph: Graph) -> None:
        """
        Updates `self.updated_graph` with nodes and edges from a second graph.

        Args:
            lookup_graph (Graph): The graph whose nodes are being evaluated and fused/inserted.
        """
        for node in lookup_graph.nodes.values():
            best_match: GraphNodeClass
            score: float

            best_match, score = self._search_best_match(node)

            if score < self._threshold:
                self._fusion_nodes(node, best_match)
                self.updated_graph.current_node = best_match
            else:
                new_node = copy.deepcopy(node)
                new_node.id = self.updated_graph.node_id

                self._include_node(new_node, best_match)

                self.updated_graph.nodes[new_node.id] = new_node
                self.updated_graph.current_node = new_node

                self.updated_graph.node_id += 1
        
        return None


    def _include_node(self, new_node: GraphNodeClass, best_match: GraphNodeClass) -> None:
        """
        Adds a new node to the graph, connects it to the current node,
        and attempts to reroute an existing neighbor edge through this node.

        Args:
            new_node (GraphNodeClass): The new node to insert into the graph.
        """

        next_node = self._find_next_node((best_match, new_node))
        self.updated_graph.edges.append((best_match.id, new_node.id))

        if next_node is not None:
            self.updated_graph.edges.append((new_node.id, next_node.id))
            # Remove direct connection from best_match to next_node if it exists

            direct_edge = (best_match.id, next_node.id)
            reverse_edge = (next_node.id, best_match.id)
            if direct_edge in self.updated_graph.edges:
                self.updated_graph.edges.remove(direct_edge)
            if reverse_edge in self.updated_graph.edges:
                self.updated_graph.edges.remove(reverse_edge)

            self._check_next_node_neighbors(next_node, new_node)

        return None


    def _find_next_node(self, new_edge: tuple[GraphNodeClass, GraphNodeClass]) -> GraphNodeClass | None:
        """
        Finds the neighbor of the current node that most closely aligns
        (via cosine similarity) with the direction of the new edge.

        Args:
            new_edge (tuple[GraphNodeClass, GraphNodeClass]): Tuple of (previous_match, new_node)

        Returns:
            Optional[GraphNodeClass]: Neighbor node to connect through, or None if none match well.
        """
        previous_match, new_node = new_edge
        direction_vector: np.ndarray = np.array(new_node.pose[:2]) - np.array(previous_match.pose[:2])

        direction_norm = np.linalg.norm(direction_vector)
        if direction_norm == 0:
            return None
        direction_vector /= direction_norm

        max_similarity: float = float("-inf")
        best_match: None | GraphNodeClass = None

        relevant_edges_idx: list[tuple[int, int]] = [e for e in self.updated_graph.edges if previous_match.id in e]
        relevant_edges: list[tuple[GraphNodeClass, GraphNodeClass]] = [
            (self.updated_graph.nodes[node_idx], self.updated_graph.nodes[adj_idx])
            for node_idx, adj_idx in relevant_edges_idx
        ]

        for edge in relevant_edges:
            neighbor: GraphNodeClass = edge[1] if edge[0] == previous_match else edge[0]
            edge_vector: np.ndarray = np.array(neighbor.pose[:2]) - np.array(previous_match.pose[:2])

            edge_norm = np.linalg.norm(edge_vector)
            if edge_norm == 0:
                continue
            edge_vector /= edge_norm

            similarity: float = np.dot(direction_vector, edge_vector) # maximum 1

            if similarity > self._similarity_threshold and similarity > max_similarity:
                max_similarity = similarity
                best_match = neighbor

        return best_match


    def _search_best_match(self, node: GraphNodeClass, k: int = 3) -> tuple[GraphNodeClass, float]:
        """
        Find the best matching node to the given node in the lookup graph using a KD-tree
        for spatial proximity and cosine distance for visual similarity.

        Args:
            node (GraphNodeClass): The query node.
            k (int): The number of nearest spatial neighbors to consider.

        Returns:
            tuple[GraphNodeClass, float]: The best matching node and the combined similarity score.
        """
        # Extract 2D poses for KD-tree
        node_positions: np.ndarray = np.array([n.pose[:2] for n in self.updated_graph.nodes.values()])
        node_list: list[GraphNodeClass] = list(self.updated_graph.nodes.values())

        kd_tree: KDTree = KDTree(node_positions)

        # Find spatially nearest neighbors
        distances, indices = kd_tree.query(node.pose[:2], k=min(k, len(node_list)))
        candidates: list[GraphNodeClass] = [node_list[i] for i in np.atleast_1d(indices)]

        best_node: GraphNodeClass | None = None
        best_score: float = float("inf")

        assert 0.0 <= self._pose_weight <= 1.0, "Pose weight must be between 0 and 1."

        for i, candidate in enumerate(candidates):
            pose_distance: float = distances[i]
            visual_distance: float = cosine(node.visual_features, candidate.visual_features)

            # Weighted sum of pose distance and visual distance
            score: float = (
                self._pose_weight * pose_distance
                + (1 - self._pose_weight) * visual_distance
            )

            if score < best_score:
                best_score = score
                best_node = candidate

        return best_node, best_score
    

    def _check_next_node_neighbors(self, next_node: GraphNodeClass, new_node: GraphNodeClass) -> None:
        """
        Attempts to insert new_node between next_node and its neighbors if it improves graph structure.

        If new_node lies in a similar direction and forms a shorter path to a neighbor of next_node,
        it replaces the direct edge with two edges: next_node <-> new_node and new_node <-> neighbor.

        Args:
            next_node (GraphNodeClass): An existing node in the graph.
            new_node (GraphNodeClass): A candidate node to insert between next_node and its neighbors.
        """
        # Now we check the next node's neighbors in case we can also introduce the new node between them.
        direction_vector: np.array = np.array(next_node.pose[:2]) - np.array(new_node.pose[:2])

        relevant_edges_idx: list[tuple[int, int]] = [e for e in self.updated_graph.edges
                                                     if next_node.id in e and new_node.id not in e]
        relevant_edges: list[tuple[GraphNodeClass, GraphNodeClass]] = [(self.updated_graph.nodes[node_idx], self.updated_graph.nodes[adj_idx])
                                                                    for node_idx, adj_idx in relevant_edges_idx]
        
        for edge in relevant_edges:
            neighbor: GraphNodeClass = edge[1] if edge[0] == next_node else edge[0]
            edge_vector: np.array = np.array(neighbor.pose[:2]) - np.array(next_node.pose[:2])
            direction_vector: np.array = np.array(neighbor.pose[:2]) - np.array(new_node.pose[:2])

            similarity: float = np.dot(direction_vector, edge_vector)

            edge_distance: float = self._compute_distance(neighbor.pose[0], neighbor.pose[1],
                                                          next_node.pose[0], next_node.pose[1])
            
            new_distance: float = self._compute_distance(neighbor.pose[0], neighbor.pose[1],
                                                          new_node.pose[0], new_node.pose[1])

            if similarity > self._similarity_threshold and edge_distance > new_distance:
                self.updated_graph.edges.append((new_node.id, neighbor.id))
                
                direct_edge = (neighbor.id, next_node.id)
                reverse_edge = (next_node.id, neighbor.id)
                if direct_edge in self.updated_graph.edges:
                    self.updated_graph.edges.remove(direct_edge)
                if reverse_edge in self.updated_graph.edges:
                    self.updated_graph.edges.remove(reverse_edge)

        return None


    def _fusion_nodes(self, node: GraphNodeClass, best_match: GraphNodeClass) -> None:
        """


        Args:
            node_1 (GraphNodeClass): Node coming from graph 2.
            node_2 (GraphNodeClass): Node coming from graph 1.

        Returns:
            GraphNodeClass: _description_
        """

        new_pose: tuple[float, float, float] = self._average_pose(node.pose, best_match.pose)
        new_image: np.ndarray = self.stitch_images(node.image, best_match.image,
                                                min_matches=self._min_matches)
        tensor_image: torch.Tensor = process_stitched_image(new_image)
        new_visual_features: np.ndarray = self._extract_features(tensor_image)

        best_match.pose = new_pose
        best_match.image = new_image
        best_match.visual_features = new_visual_features

        best_match.update_semantics()

        return None
    

    def _extract_features(self, image: torch.Tensor) -> np.ndarray:
        """
        

        Args:
            image (torch.Tensor): _description_

        Returns:
            np.ndarray: _description_
        """
        features: torch.Tensor = self._camera.extract_features(image)

        if features.is_cuda:
            features: list = features.view(-1).cpu().tolist()
        else:
            features: list = features.view(-1).tolist()

        return np.array(features).astype("float32")
    

    def generate_map(self) -> None:
        """
        Generates final path image, drawing both node positions and edges.
        """
        map_folder: str = os.path.join("images/maps", self._map_name)
        file_name: str = self._trajectory + ".png"
        output_path: str = os.path.join(f"images/final_aligned_maps/{self._map_name[:-4]}", file_name)

        map_img = cv2.imread(map_folder)

        # Draw nodes
        for node in self.updated_graph.nodes.values():
            y, x, _ = node.pose  # Assuming pose = (y, x, theta)
            px, py = world_to_pixel(-x, y, map_img.shape, self._world_limits, origin=self._origin)
            cv2.circle(map_img, (px, py), 5, (0, 0, 255), -1)

        # Draw edges
        for idx_1, idx_2 in self.updated_graph.edges:
            node_1: GraphNodeClass = self.updated_graph.nodes.get(idx_1)
            node_2: GraphNodeClass = self.updated_graph.nodes.get(idx_2)
            if node_1 is not None and node_2 is not None:
                y1, x1, _ = node_1.pose
                y2, x2, _ = node_2.pose
                p1 = world_to_pixel(-x1, y1, map_img.shape, self._world_limits, origin=self._origin)
                p2 = world_to_pixel(-x2, y2, map_img.shape, self._world_limits, origin=self._origin)
                cv2.line(map_img, p1, p2, (0, 255, 0), 2)  # Green lines for edges

        cv2.imwrite(output_path, map_img)

        return None

    

    def _average_pose(self, pose_1: tuple[float, float, float],
                      pose_2: tuple[float, float, float]) -> tuple[float, float, float]:
        """
        

        Args:
            pose_1 (tuple[float, float, float]): _description_
            pose_2 (tuple[float, float, float]): _description_

        Returns:
            tuple[float, float, float]: _description_
        """
        return tuple((a + b) / 2 for a, b in zip(pose_1, pose_2))
    

    def _compute_distance(self, x: float, y: float, prev_x: float, prev_y: float) -> float:
        """
        Computes Euclidean distance between two points.

        Args:
            x (float): current x coordinate.
            y (float): current y coordinate.
            prev_x (float): previous x coordinate.
            prev_y (float): previous y coordinate.

        Returns:
            float: distance between (x, y) and (prev_x, prev_y).
        """
        return np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
    

    def stitch_images(
        self,
        image_1: np.ndarray,
        image_2: np.ndarray,
        min_matches: int = 4
    ) -> np.ndarray:
        """
        Stitches two images together using SIFT feature matching and homography.
        
        Args:
            image_1 (np.ndarray): First input image (BGR format).
            image_2 (np.ndarray): Second input image (BGR format).
            min_matches (int): Minimum number of good matches to attempt stitching.
        
        Returns:
            np.ndarray: Stitched image if successful, otherwise fallback image.
        """

        gray_1: np.ndarray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        gray_2: np.ndarray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

        sift: cv2.SIFT = cv2.SIFT_create()
        kp1: list[cv2.KeyPoint]
        des1: np.ndarray
        kp1, des1 = sift.detectAndCompute(gray_1, None)

        kp2: list[cv2.KeyPoint]
        des2: np.ndarray
        kp2, des2 = sift.detectAndCompute(gray_2, None)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return image_1  # No descriptors or too few: can't proceed

        index_params: dict[str, int] = {"algorithm": 1, "trees": 5}
        search_params: dict[str, int] = {"checks": 50}
        flann: cv2.FlannBasedMatcher = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            matches: list[list[cv2.DMatch]] = flann.knnMatch(des1, des2, k=2)
        except cv2.error:
            return image_1  # Matching failed

        good_matches: list[cv2.DMatch] = [
            m for m, n in matches if m.distance < 0.7 * n.distance
        ]

        if len(good_matches) < min_matches:
            return concat_images(image_1, image_2)  # Low match count, maybe no overlap

        src_pts: np.ndarray = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts: np.ndarray = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H: np.ndarray
        mask_homography: np.ndarray
        H, mask_homography = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None or H.shape != (3, 3):
            return image_1  # Homography computation failed

        height: int
        width: int
        height, width, _ = image_2.shape
        warped_img1: np.ndarray = cv2.warpPerspective(image_1, H, (width * 2, height))

        # Overlay image_2 onto the warp
        warped_img1[0:height, 0:width] = image_2

        # Basic content quality check
        gray_warped: np.ndarray = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY)
        _, mask_thresh = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)
        nonzero_ratio: float = np.count_nonzero(mask_thresh) / (mask_thresh.shape[0] * mask_thresh.shape[1])

        if nonzero_ratio < 0.2:
            return image_1  # Warped result is mostly empty

        stitched_image: np.ndarray = self.crop_black_borders(warped_img1, mask_thresh)

        return stitched_image

    def crop_black_borders(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Crops black borders from an image using a mask.

        Args:
            image (np.ndarray): stitched image with potential black borders.
            mask (np.ndarray): binary mask indicating valid regions.

        Returns:
            np.ndarray: cropped image without black borders.
        """
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y+h, x:x+w]