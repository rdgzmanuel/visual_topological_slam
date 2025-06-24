import numpy as np
import cv2
import os
import rclpy.logging
import torch
import math
from torchvision import transforms
from typing import Optional
from sklearn.metrics.pairwise import cosine_distances
import gtsam
from gtsam import Pose2, BetweenFactorPose2, noiseModel
from PIL import Image
from vts_graph_building.node import GraphNodeClass
from vts_camera.camera import Camera


class GraphBuilder:
    def __init__(self, n: int, gamma_proportion: float, delta_proportion: float, distance_threshold: float,
                 initial_pose: tuple[float, float, float], world_limits: tuple[float, float, float, float],
                 map_name: str, origin: tuple[int, int], weights: tuple[float], trajectory: str,
                 model_name: str, ext_rewiring: bool):
        """
        Constructor of the GraphBuilder class. Contains relevant graph information
        and hyperparameters.

        Args:
            initial_pose (tuple[float, float, float], optional): initial pose of the
            system. Defaults to (0.0, 0.0, 0.0).
        """
        self._initial_pose: tuple[float, float, float] = initial_pose
        self.current_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.current_node: Optional[GraphNodeClass] = None
        self._current_image: Optional[np.ndarray] = None
        self.steps: int = 0
        self.graph: list[tuple[GraphNodeClass, GraphNodeClass]] = []
        self.window_full: bool = False

        self._images_pose: list[tuple[np.ndarray, tuple[float, float, float], np.ndarray]] = []
        self.window_images: Optional[np.ndarray] = None
        self.eigenvalues: list[float] = []
        self._representative_candidates: list[tuple[int, float]] = []

        self._world_limits: tuple[float, float, float, float] = world_limits
        self._map_name: str = map_name
        self._origin: tuple[int, int] = origin
        self._weights: tuple[float] = weights

        self._trajectory: str = trajectory
        self._images_path: str = os.path.join("../../project/seq_data/", self._trajectory)
        self._images_path = os.path.join(self._images_path, "std_cam")

        self._n: int = n
        self._lambda_2_max: float = n / (n - 1)
        self._gamma: float = gamma_proportion * self._lambda_2_max
        self._delta: float = delta_proportion * self._gamma
        self._distance_threshold: float = distance_threshold

        self._max_similarity: float = 0
        self._max_index: int = 0

        self._current_alg_conenctivity: float = 0.0
        self._node_id = 0

        # peak and valley
        self.look_for_maximum: bool = True
        self.max_value: float = float("-inf")
        self.min_value: float = float("inf")
        self.min_idx: int = 0
        self._current_image_idx: int = -1
        self._max_size: int = 100

        # images stitching
        self._min_matches: int = 4
        self._min_descriptors: int = 2
        self._camera: Camera = Camera(model_name)
        self._image_shape: tuple[int, int, int] | None = None

        # rewiring
        self._ext_rewiring: bool = ext_rewiring
        self._rewiring_threshold: float = 3.0 # 3.0 fA 1.25 sA   4.0 fE  4.25 sE
        self._external_rewiring_threshold: float = 2.5 # 2.5 fA 1.0 sA   3.5 fE   4.0 sE
        self._hard_threshold: float = 0.7 # 0.7 fAE  0.55 sA 
        self._min_rewire_nodes: int = 3

        self._logger = rclpy.logging.get_logger('GraphBuilder')


    def update_pose(self, v: float, w: float, time_difference: float) -> None:
        """
        Updates current pose of the system with odometry measurements.

        Args:
            v (float): Linear velocity.
            w (float): Angular velocity.
            time_difference (float): Time difference between previous and current pose.
        """
        if self.steps == 0:
            prev_x, prev_y, prev_theta = self._initial_pose
        else:
            prev_x, prev_y, prev_theta = self.current_pose

        x: float = prev_x + v * np.cos(prev_theta + (w * time_difference) / 2) * time_difference
        y: float = prev_y + v * np.sin(prev_theta + (w * time_difference) / 2) * time_difference
        theta: float = prev_theta + w * time_difference

        # Normalize theta to be in the range [-π, π]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        self.current_pose = (x, y, theta)
        self.steps += 1

        self._plot_node_on_map(self.current_pose, node=False)

        return None


    def new_update_pose(self, image_name: str) -> None:
        """
        Upgrades system's pose using directly images names and performing a transformaton.

        Args:
            image_name (str): _description_
        """
        self._current_image: np.ndarray = self._convert_image(image_name)

        splitted_msg: list[str] = image_name.split("_")
        x: float = float(splitted_msg[1][1:])
        x = self._weights[0] * (x**5) + self._weights[1] * (x**4) + self._weights[2] * (x**3)\
            + self._weights[3] * (x**2) + self._weights[4] * x + self._weights[5]
        y: float = float(splitted_msg[2][1:])
        y = self._weights[6] * (y**5) + self._weights[7] * (y**4) + self._weights[8] * (y**3)\
            + self._weights[9] * (y**2) + self._weights[10] * y + self._weights[11]
        # self._logger.warn("pre norml")
        theta: float = self._normalize_angle(float(splitted_msg[3][1:5]))

        self.current_pose = (x, y, theta)
        # self._logger.warn("pre plot")
        self._plot_node_on_map(self.current_pose, node=False)
        # self._logger.warn("post plot")
        self.steps += 1


    def update_pose_odom(self, image_name: str, new_pose: tuple[float, float, float]) -> None:
        """
        Upgrades system's pose using directly images names and performing a transformaton.

        Args:
            image_name (str): _description_
        """
        self._current_image: np.ndarray = self._convert_image(image_name)
        # self._logger.warn("post convrt")

        x: float
        y: float
        theta: float
        x, y, theta = new_pose

        x = self._weights[0] * (x**5) + self._weights[1] * (x**4) + self._weights[2] * (x**3)\
            + self._weights[3] * (x**2) + self._weights[4] * x + self._weights[5]

        y = self._weights[6] * (y**5) + self._weights[7] * (y**4) + self._weights[8] * (y**3)\
            + self._weights[9] * (y**2) + self._weights[10] * y + self._weights[11]

        theta = self._normalize_angle(float(theta))

        self.current_pose = (x, y, theta)
        # self._logger.warn("pre plot")
        self._plot_node_on_map(self.current_pose, node=False)
        # self._logger.warn("post plot")
        self.steps += 1


    def _convert_image(self, image_name: str) -> np.ndarray:
        """
        Converts image file into np.ndarray object.

        Args:
            image_name (str): name of the image.

        Returns:
            np.ndarray: np.ndarray representation of the image.
        """
        image_path: str = os.path.join(self._images_path, image_name)
        image: np.ndarray = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if self._current_image is None and image is not None:
            # self._logger.warn(f"path {image_path}")
            self._image_shape = image.shape
        # self._logger.warn("pre resize")

        if self._image_shape is not None:
            image = cv2.resize(image, (self._image_shape[1], self._image_shape[0]))

        return image


    def upgrade_pose_odom(self, image_name: str) -> None:
        """
        Calculates position from images name using odometry: storing the previous pose and estimating
        linear and angular speeds from change in position. Speeds are then used to update current pose.
        DOESN'T WORK: timestamp difference is too little that linear spped (v) diverges.

        Args:
            image_name (str): _description_
        """

        splitted_msg: list[str] = image_name.split("_")
        timestamp: float = float(splitted_msg[0][1:])
        x: float = float(splitted_msg[1][1:])
        x = self._weights[0] * (x**5) + self._weights[1] * (x**4) + self._weights[2] * (x**3)\
            + self._weights[3] * (x**2) + self._weights[4] * x + self._weights[5]
        y: float = float(splitted_msg[2][1:])
        y = self._weights[6] * (y**5) + self._weights[7] * (y**4) + self._weights[8] * (y**3)\
            + self._weights[9] * (y**2) + self._weights[10] * y + self._weights[11]

        theta: float = self._normalize_angle(float(splitted_msg[3][1:5]))

        if self.steps == 0:
            prev_x, prev_y, prev_theta = self._initial_pose
            self.current_pose = (prev_x, prev_y, prev_theta)
        else:
            prev_x, prev_y, prev_theta = self.current_pose
            time_difference: float = self._prev_timestamp - timestamp

            distance: float = self._compute_distance(x, y, prev_x, prev_y)
            v: float = distance / time_difference

            delta_theta: float = self._normalize_angle(theta - prev_theta)
            w: float = delta_theta / time_difference

            x: float = prev_x + v * np.cos(prev_theta + (w * time_difference) / 2) * time_difference
            y: float = prev_y + v * np.sin(prev_theta + (w * time_difference) / 2) * time_difference
            theta: float = self._normalize_angle(prev_theta + w * time_difference)

            self.current_pose = (x, y, theta)

        self._plot_node_on_map(self.current_pose, node=False)

        self._prev_timestamp = timestamp
        self.steps += 1
    

    def _compute_distance(self, x: float, y: float, prev_x: float, prev_y: float) -> float:
        """
        Computes Euclidean distance between two points.

        Args:
            x (float): Current x coordinate.
            y (float): Current y coordinate.
            prev_x (float): Previous x coordinate.
            prev_y (float): Previous y coordinate.

        Returns:
            float: Distance between (x, y) and (prev_x, prev_y).
        """
        return math.hypot(x - prev_x, y - prev_y)


    def _normalize_angle(self, angle: float) -> float:
        """
        Noralizes given angle.

        Args:
            angle (float): angle to normalize.

        Returns:
            float: normalized angle.
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi


    def _build_affinity_matrix(self) -> None:
        """
        Adds empty row and column to affinity matrix
        """
        if not hasattr(self, "affinity") or self.affinity is None:
            self.affinity: np.ndarray = np.zeros((1, 1))
        else:
            rows: int
            columns: int
            rows, columns = self.affinity.shape

            new_affinity: np.ndarray = np.zeros((rows + 1, columns + 1))
            new_affinity[:rows, :columns] = self.affinity

            self.affinity = new_affinity

        return None
    

    def _move_window(self) -> None:
        """
        Moves window removing first row and column from affinity matrix and first image from _window_images.
        """
        self.affinity[:-1, :-1] = self.affinity[1:, 1:]
        self.affinity[-1, :] = 0
        self.affinity[:, -1] = 0
        self.window_images = self.window_images[1:]

        return None


    def _compute_similarities(self, array_data: np.ndarray) -> None:
        """
        Updates affinity matrix computing the similarities with previous images from the
        window and current incoming image.

        Args:
            idx (int): _description_
        """

        similarities: np.ndarray = np.dot(self.window_images, array_data)

        self.affinity[:-1, -1] = similarities[:-1]
        self.affinity[-1, :-1] = similarities[:-1]
        self.affinity[-1, -1] = 0.0

        return None
    

    def _update_degree_matrix(self) -> None:
        """
        Updates degree matrix with the new affinity matrix. Also looks for a new maximum
        in the similarity score.
        """
        self.degree: np.ndarray = np.diag(self.affinity.sum(axis=1))
        max_idx: int = np.argmax(np.diag(self.degree))

        if np.diag(self.degree)[max_idx] > self._max_similarity:
            self._max_similarity = np.diag(self.degree)[max_idx]
            self._max_index = max_idx
        
        return None


    def _update_laplacian_sym_matrix(self) -> None:
        """
        Computes the Laplacian matrix using the affinity and degree matrices.
        """
        degree_inv_sqrt: np.ndarray = np.diag(1.0 / np.sqrt(np.diag(self.degree)))

        self._laplacian_sym: np.ndarray = np.eye(self.affinity.shape[0]) - degree_inv_sqrt @\
        self.affinity @ degree_inv_sqrt

        return None


    def update_matrices(self, array_data: np.ndarray) -> None:
        """
        Updates affinity, degree and Laplacian matrices with the new image.
        Args:
            array_data (np.ndarray): _description_
        """
        norm_array_data: np.ndarray = array_data / np.linalg.norm(array_data)
        self._images_pose.append((norm_array_data, self.current_pose, self._current_image))

        if len(self._images_pose) > self._max_size:
            self._images_pose.pop(0)

        self._current_image_idx += 1

        if self.window_images is None:
            self.window_images = np.empty((0, norm_array_data.shape[0]))
        self.window_images = np.vstack([self.window_images, norm_array_data])

        if len(self.window_images) <= self._n:
            # still building affinity matrix
            self._build_affinity_matrix()
        else:
            # affinity matrix already built, need to move window
            self._move_window()

        self._compute_similarities(norm_array_data)
        self._update_degree_matrix()
        self._update_laplacian_sym_matrix()

        # self._logger.warn(f"Size: {sys.getsizeof(self._images_pose)} len: {len(self._images_pose)}")
        self.window_full = len(self.window_images) == self._n

        return None


    def _obtain_eigenvalue(self) -> float:
        """
        Return second lowest eigenvalue of the Laplacian matrix

        Returns:
            float: _description_
        """
        eigenvalues: np.ndarray
        eigenvalues, _ = np.linalg.eig(self._laplacian_sym)
        eigenvalues = np.sort(eigenvalues)

        lambda_2: float = eigenvalues[1]
        self.eigenvalues.append(lambda_2)

        return lambda_2


    def look_for_valley(self) -> int:
        """
        Applies peak and valley finding algorithm.

        Returns:
            bool: _description_
        """
        # current_image_idx: int = len(self._images_pose) - 1

        lambda_2: float = self._obtain_eigenvalue()

        valley_idx: int = 0

        if lambda_2 > self.max_value:
            self.max_value = lambda_2
        elif lambda_2 < self.min_value:
            self.min_value = lambda_2
            self.min_idx = self._current_image_idx
        
        if self.look_for_maximum:
            if self._current_alg_conenctivity < (self.max_value - self._delta) and self.max_value >= self._gamma:
                self.look_for_maximum = False
                self.min_value = lambda_2
        else:
            if self._current_alg_conenctivity > (self.min_value + self._delta):
                self.look_for_maximum = True
                self.max_value = lambda_2
                valley_idx = self.min_idx
        
        self._current_alg_conenctivity = lambda_2
        
        return lambda_2, valley_idx


    def update_graph(self) -> None:
        """
        Creates a new node with selected representative and updates relevant variables.

        Args:
            valley_idx (int): _description_
        """
        idx: int = len(self._images_pose) - len(self.window_images) + self._max_index
        representative: tuple[np.ndarray, tuple[float, float, float], np.ndarray] = self._images_pose[idx]
        self._max_similarity = float("-inf")

        new_node: GraphNodeClass = GraphNodeClass(id=self._node_id, pose=representative[1],
                                            visual_features=representative[0], image=representative[2])

        if self.current_node is None:
            self._node_id += 1
            new_node.update_semantics()
            # self._logger.warn(f"New embeddings: {new_node.semantics}")
            self.current_node = new_node
            # self._logger.warn(f"NEW CURRENT INITIAL UPDATE {self.current_node.pose}")
            # self._plot_node_on_map(self.current_node.pose)

        else:
            closest_neighbor: GraphNodeClass = self._search_closest_neighbor(new_node.pose, new_node.visual_features)
            distance: float = self._compute_distance(closest_neighbor.pose[0], closest_neighbor.pose[1],
                                                     new_node.pose[0], new_node.pose[1])
            # self._logger.warn(f"closest {closest_neighbor.pose}, distance {distance}")
            if distance < self._distance_threshold:
                # Revisit confirmed
                closest_neighbor = self._fusion_nodes(new_node, closest_neighbor)
                loop_nodes: list[GraphNodeClass] = self._obtain_loop_nodes(closest_neighbor)
                # self._logger.warn(f"Loop nodes {[node.id for node in loop_nodes]}")
                relevant_edges: list[tuple[GraphNodeClass, GraphNodeClass]] = self._obtain_relevant_edges(loop_nodes)
                
                self._optimize_loop_poses(loop_nodes, relevant_edges)
                # self._logger.warn(f"Opitimizing time {time.time() - start}")
                self._rewire_graph(loop_nodes, relevant_edges, self._rewiring_threshold)
                # self._logger.warn(f"NEW CURRENT FUSED AND REWIRED CURRENT {self.current_node.pose} -> NN {closest_neighbor.pose}")
                self.current_node = closest_neighbor
                
                # self._logger.warn(f"Finsished rewiring {[(node.id, adj.id) for node, adj in self.graph]}")
                # self._logger.warn(f"Poses {[(node.id, node.pose) for node, _ in self.graph]}")

            else:
                self._node_id += 1
                new_node.update_semantics()
                                   
                # self._logger.warn(f"add {self.current_node.pose[:2]} {new_node.pose[:2]}")
                self.graph.append((self.current_node, new_node))

                if self._ext_rewiring:
                    # Before adding a new node, check current node's neighbors edges to see if we can project it
                    relevant_edges: list[tuple[GraphNodeClass, GraphNodeClass]] = [edge for edge in self.graph if self.current_node in edge] 
                    self._rewire_graph([new_node], relevant_edges, self._external_rewiring_threshold)

                    # Before adding a new edge, check current node's neighbors to see if we can project them
                    relevant_nodes: list[GraphNodeClass] = list(self.current_node.neighbors)
                    self._rewire_graph(relevant_nodes, [(self.current_node, new_node)], self._external_rewiring_threshold)

                new_node.neighbors.add(self.current_node)
                self.current_node.neighbors.add(new_node)
                self.current_node = new_node
                # self._logger.warn(f"NEW CURRENT {self.current_node.pose}")
                # self._plot_node_on_map(self.current_node.pose)
                # self._logger.warn("finish update")
            # self._logger.warn(f"----")

        return None
    

    def _search_closest_neighbor(self, pose: tuple[float, float, float],
                                new_visual_features: Optional[np.ndarray] = None) -> GraphNodeClass:
        """
        Finds the closest neighbor to the given pose, using both position and visual similarity.

        Args:
            pose (tuple[float, float, float]): Target (x, y, theta) pose.
            new_visual_features (Optional[np.ndarray]): Visual features of the current node.

        Returns:
            GraphNodeClass: Closest graph node.
        """
        if not self.graph:
            return self.current_node

        unique_nodes: set[GraphNodeClass] = set()
        for node_a, node_b in self.graph:
            unique_nodes.update([node_a, node_b])

        nodes: list[GraphNodeClass] = list(unique_nodes)
        positions: np.ndarray = np.array([node.pose[:2] for node in nodes])
        target_position: np.ndarray = np.array(pose[:2])
        position_distances: np.ndarray = np.linalg.norm(positions - target_position, axis=1)

        if new_visual_features is not None:
            angles: np.ndarray = np.array([node.pose[2] for node in nodes])
            angle_diff: np.ndarray = np.abs((angles - pose[2] + np.pi) % (2 * np.pi) - np.pi)  # [0, pi]
            visual_weights: np.ndarray = 0.5 * (1 - angle_diff / np.pi)  # [0, 0.5]

            visual_features: np.ndarray = np.array([node.visual_features for node in nodes])
            visual_similarities: np.ndarray = cosine_distances(visual_features, new_visual_features[np.newaxis, :]).flatten()

            overall_similarities: np.ndarray = (1 - visual_weights) * position_distances + visual_weights * visual_similarities
        else:
            overall_similarities: np.ndarray = position_distances

        best_idx: int = np.argmin(overall_similarities)
        return nodes[best_idx]
    

    def _fusion_nodes(self, new_node: GraphNodeClass, closest_neighbor: GraphNodeClass) -> GraphNodeClass:
        """
        Fusions new node with its closests neighbor, stitching their images and feeding the
        result to the model to obtain new features.

        Args:
            new_node (GraphNodeClass): _description_
            closets_neighbor (GraphNodeClass): _description_
        """
        # new_pose = self._average_pose(closest_neighbor.pose, new_node.pose)
        new_image: np.ndarray = self.stitch_images(closest_neighbor.image, new_node.image,
                                            min_matches=self._min_matches)
        new_image = cv2.resize(new_image, (self._image_shape[1], self._image_shape[0]))
        tensor_image: torch.Tensor = process_stitched_image(new_image)
        new_visual_features: np.ndarray = self._extract_features(tensor_image)

        # closest_neighbor.pose = new_pose
        closest_neighbor.image = new_image
        closest_neighbor.visual_features = new_visual_features
        # self._plot_node_on_map(closest_neighbor.pose)

        closest_neighbor.update_semantics()

        if self.current_node != closest_neighbor:
            closest_neighbor.neighbors.add(self.current_node)
            self.current_node.neighbors.add(closest_neighbor)
            new_edge: tuple[GraphNodeClass, GraphNodeClass] = (self.current_node, closest_neighbor)
            if new_edge not in self.graph: # and (new_edge[1], new_edge[0]) not in self.graph:
                self.graph.append(new_edge)
                # self._logger.warn(f"Add {new_edge[0].pose[:2]} {new_edge[1].pose[:2]}")

        return closest_neighbor
    

    def _average_pose(self, pose_1: tuple[float, float, float],
                      pose_2: tuple[float, float, float]) -> tuple[float, float, float]:
        """
        

        Args:
            pose_1 (tuple[float, float, float]): _description_
            pose_2 (tuple[float, float, float]): _description_

        Returns:
            tuple[float, float, float]: _description_
        """
        new_pose: list[float, float, float] = [(a + b) / 2 for a, b in zip(pose_1, pose_2)]
        new_pose[2] = self._normalize_angle(new_pose[2])
        return tuple(new_pose)


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
    

    def _obtain_loop_nodes(self, closest_node: GraphNodeClass) -> list[GraphNodeClass]:
        """
        Returns all nodes between current node and closest neighbor (nodes in the loop).

        Args:
            closest_node (GraphNodeClass): _description_

        Returns:
            list[GraphNodeClass]: _description_
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

        loop_nodes: list[GraphNodeClass] = list(loop_nodes)
        # self._logger.warn(f"LOOP NODES {[node.id for node in loop_nodes]}")
        # self._logger.warn(f" NODES {[(node.id, adj.id) for node, adj in self.graph]}")
        return loop_nodes
    

    def _obtain_relevant_edges(self, loop_nodes: list[GraphNodeClass]) -> list[tuple[GraphNodeClass, GraphNodeClass]]:
        """
        

        Args:
            loop_nodes (list[GraphNodeClass]): _description_

        Returns:
            list[tuple[GraphNodeClass, GraphNodeClass]]: _description_
        """
        return [edge for edge in self.graph if edge[0] in loop_nodes and edge[1] in loop_nodes]


    def _rewire_graph(self, loop_nodes: list[GraphNodeClass],
                      relevant_edges: list[tuple[GraphNodeClass, GraphNodeClass]],
                      threshold: float) -> None:
        """
        
        """
        if len(loop_nodes) > self._min_rewire_nodes:
            for node in loop_nodes:
                x, y, _ = node.pose
                projections: dict[tuple[GraphNodeClass, GraphNodeClass],
                                tuple[float, float]] = self._get_projections(node, relevant_edges)
                for edge, projection in projections.items():
                    if projection is not None:
                        distance: float = self._compute_distance(x, y, projection[0], projection[1])
                        # self._logger.warn(f"Node: {node}, proj: {edge[0].id}, {edge[1].id}, dist {distance}")
                        if distance < threshold:
                            # self._logger.warn("rewiring")
                            # self._logger.warn(f"Add {edge[0].pose[:2]} {node.pose[:2]} {edge[1].pose[:2]}")
                            # rewired edge found
                            self.graph.append((edge[0], node))
                            self.graph.append((node, edge[1]))
                            edge[0].neighbors.add(node)
                            edge[1].neighbors.add(node)
                            node.neighbors.add(edge[0])
                            node.neighbors.add(edge[1])

                            # self._logger.warn("pre aggregate")
                            self._aggregate_node(node, projection)
                            # self._logger.warn("post aggregate")
                            if edge in self.graph:
                                # self._logger.warn(f"Removes {edge[0].pose[:2]} {edge[1].pose[:2]}")
                                self.graph.remove(edge)
                                if (edge[1], edge[0]) in self.graph:
                                    self.graph.remove((edge[1], edge[0]))
                                    if edge[1] in edge[0].neighbors:
                                        edge[0].neighbors.remove(edge[1])
                                    if edge[0] in edge[1].neighbors:
                                        edge[1].neighbors.remove(edge[0])


    def _get_projections(self, 
                     node: GraphNodeClass, 
                     relevant_edges: list[tuple[GraphNodeClass, GraphNodeClass]]
                     ) -> dict[tuple[GraphNodeClass, GraphNodeClass], tuple[float, float]]:
        """
        Computes valid projections of the node onto the given edges.

        Args:
            node (GraphNodeClass): Node to project.
            relevant_edges (List[Tuple[GraphNodeClass, GraphNodeClass]]): List of graph edges.

        Returns:
            Dict[Tuple[GraphNodeClass, GraphNodeClass], Tuple[float, float]]: 
                Dictionary mapping valid edges to the (x, y) projection coordinates.
        """
        projections: dict[tuple[GraphNodeClass, GraphNodeClass], tuple[float, float]] = {}

        for edge in relevant_edges:
            n1, n2 = edge

            if node != n1 and node != n2:
                p1: np.ndarray = np.array(n1.pose[:2])
                p2: np.ndarray = np.array(n2.pose[:2])

                projection: tuple[float, float] = self._project_node(node, edge)
                proj: np.ndarray = np.array(projection)

                # Compute vectors
                edge_vector: np.ndarray = p2 - p1
                proj_vector: np.ndarray = proj - p1

                edge_length_sq: float = np.dot(edge_vector, edge_vector)

                if edge_length_sq == 0.0:
                    continue  # Avoid division by zero if nodes are at the same position

                # Compute t: where proj falls between p1 and p2
                t: float = np.dot(proj_vector, edge_vector) / edge_length_sq

                # self._logger.warn(f"Projection parameter t = {t:.5f}")

                # Check if projection falls within the edge segment
                if 0.0 <= t <= 1.0:
                    # self._logger.warn(f"Valid projection on edge {edge}")
                    projections[edge] = (proj[0], proj[1])

        return projections


    def _project_node(self, node: GraphNodeClass, edge: tuple[GraphNodeClass, GraphNodeClass]) -> tuple[float, float]:
        """
        Projects node into edge.

        Args:
            node (GraphNodeClass): The node to be projected.
            edge (tuple[GraphNodeClass, GraphNodeClass]): The two nodes that define the edge.

        Returns:
            tuple[float, float]: Coordinates of the projection point.
        """
        x1, y1, _ = edge[0].pose
        x2, y2, _ = edge[1].pose
        x, y, _ = node.pose

        v_x, v_y = x2 - x1, y2 - y1  # Edge vector (v)
        u_x, u_y = x - x1, y - y1    # Vector from first node to the node (u)

        dot_product: float = u_x * v_x + u_y * v_y
        
        v_squared_magnitude: float = v_x ** 2 + v_y ** 2
        
        projection_scalar: float = dot_product / (v_squared_magnitude + 1e-6)
        
        proj_x: float = x1 + projection_scalar * v_x
        proj_y: float = y1 + projection_scalar * v_y
        
        return proj_x, proj_y
    

    def _optimize_loop_poses(self, nodes: list[GraphNodeClass],
        graph_edges: list[tuple[GraphNodeClass, GraphNodeClass]]) -> None:
        """
        Applies pose graph optimization to correct loop closure drift.
        The first node in `nodes` is assumed to be the fixed reference (corridor entry).

        Args:
            nodes (List[GraphNodeClass]): All nodes in the loop, including the fixed corridor node.
            graph_edges (List[Tuple[GraphNodeClass, GraphNodeClass]]): List of relative pose constraints between nodes.
        """
        graph: gtsam.NonlinearFactorGraph = gtsam.NonlinearFactorGraph()
        initial_estimates: gtsam.Values = gtsam.Values()

        # Add a prior to fix the first node (corridor node).
        if len(nodes) > 0:
            fixed_node: GraphNodeClass = nodes[0]
            fixed_pose: Pose2 = Pose2(*fixed_node.pose)
            prior_noise = noiseModel.Diagonal.Sigmas([1e-6, 1e-6, 1e-6])
            graph.add(gtsam.PriorFactorPose2(fixed_node.id, fixed_pose, prior_noise))
            initial_estimates.insert(fixed_node.id, fixed_pose)

            # Add relative pose constraints (edges).
            for node1, node2 in graph_edges:
                pose1: Pose2 = Pose2(*node1.pose)
                pose2: Pose2 = Pose2(*node2.pose)
                relative_pose: Pose2 = pose1.between(pose2)
                model = noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1])
                graph.add(BetweenFactorPose2(node1.id, node2.id, relative_pose, model))

                # Insert initial guesses if not already present.
                if not initial_estimates.exists(node1.id):
                    initial_estimates.insert(node1.id, pose1)
                if not initial_estimates.exists(node2.id):
                    initial_estimates.insert(node2.id, pose2)

            # Optimize the pose graph.
            params: gtsam.LevenbergMarquardtParams = gtsam.LevenbergMarquardtParams()
            optimizer: gtsam.LevenbergMarquardtOptimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
            result: gtsam.Values = optimizer.optimize()

            # Update the node poses with the optimized result.
            for node in nodes:
                if result.exists(node.id):
                    optimized_pose: Pose2 = result.atPose2(node.id)
                    node.pose = (optimized_pose.x(), optimized_pose.y(), optimized_pose.theta())

        return None
    

    def _aggregate_node(self, node: GraphNodeClass, projection: tuple[float, float]) -> None:
        """
        

        Args:
            node (GraphNodeClass): _description_
            projection (tuple[float, float]): _description_
        """

        new_projection_pose: tuple[float, float, float] = (projection[0],
                                                      projection[1], self._normalize_angle(node.pose[2] + np.pi))
        new_pose: tuple[float, float, float] = self._average_pose(node.pose, new_projection_pose)

        node.pose = new_pose

        image: np.ndarray = self._find_closest_image(projection)

        new_image: np.ndarray = self.stitch_images(node.image, image,
                                            min_matches=self._min_matches)
        new_image = cv2.resize(new_image, (self._image_shape[1], self._image_shape[0]))
        tensor_image: torch.Tensor = process_stitched_image(new_image)
        new_visual_features: np.ndarray = self._extract_features(tensor_image)

        node.image = new_image
        node.visual_features = new_visual_features

        node.update_semantics()

        return None


    def _find_closest_image(self, query_coords: tuple[float, float]) -> np.ndarray:
        """
        Returns the image with the closest 2D position to the given coordinates.

        Args:
            query_coords: (x, y) tuple representing the target coordinates.

        Returns:
            The image (np.ndarray) whose (x, y) pose is closest to query_coords.
        """
        query: np.ndarray = np.array(query_coords)

        positions: np.ndarray = np.array([pose[1][:2] for pose in self._images_pose])

        deltas: np.ndarray = positions - query
        squared_distances: np.ndarray = np.einsum('ij,ij->i', deltas, deltas)

        min_idx: int = np.argmin(squared_distances)

        return self._images_pose[min_idx][2]
    

    def check_pose(self) -> None:
        """
        Checks whether the robot's current pose corresponds to an already existing node
        in the graph. If a nearby node is found within a hard distance threshold, a loop
        closure is detected. The graph is then rewired to improve connectivity and reduce
        odometry drift. The matched node is updated with new visual information.
        """
        x: float
        y: float
        theta: float
        x, y, theta = self.current_pose

        match: Optional[GraphNodeClass] = self._search_closest_neighbor((x, y, theta))

        if match != self.current_node:

            distance: float = self._compute_distance(x, y, match.pose[0], match.pose[1])
            
            if distance < self._hard_threshold:
                loop_nodes: list[GraphNodeClass] = self._obtain_loop_nodes(match)
                relevant_edges: list[tuple[GraphNodeClass, GraphNodeClass]] = self._obtain_relevant_edges(loop_nodes)
                self._rewire_graph(loop_nodes, relevant_edges, self._rewiring_threshold)

                new_edge: tuple[GraphNodeClass, GraphNodeClass] = (self.current_node, match)
                match.neighbors.add(self.current_node)
                self.current_node.neighbors.add(match)

                if new_edge not in self.graph:
                    self.graph.append(new_edge)
                
                self.current_node = match

                new_image: np.ndarray = self.stitch_images(
                    self.current_node.image,
                    self._current_image,
                    min_matches=self._min_matches
                )
                new_image = cv2.resize(new_image, (self._image_shape[1], self._image_shape[0]))
                tensor_image: torch.Tensor = process_stitched_image(new_image)
                new_visual_features: np.ndarray = self._extract_features(tensor_image)

                self.current_node.image = new_image
                self.current_node.visual_features = new_visual_features
                self.current_node.update_semantics()

        return None


    def _plot_node_on_map(self, pose: tuple[float, float, float], node=True) -> None:
        """
        Plots SLAM node on a given map image.

        Args:
            pose (tuple[float, float, float]): _description_
        """
        map_folder: str = os.path.join("images/maps", self._map_name)
        output_path: str = self._trajectory + "_nodes.png"
        output_path = os.path.join(f"images/running_maps/{self._map_name[:-4]}", output_path)

        if self.steps != 0:
            map_folder = output_path

        map_img = cv2.imread(map_folder)

        y, x, _ = pose  # Ignore theta for now
        px, py = self.world_to_pixel(-x, y, map_img.shape, self._world_limits, self._origin)
        # self._logger.warn("post world")
        if node:
            cv2.circle(map_img, (px, py), 5, (0, 0, 255), -1)
        else:
            cv2.circle(map_img, (px, py), 1, (255, 0, 0), -1)
        # self._logger.warn(f"pre write {output_path}")
        cv2.imwrite(output_path, map_img)
        # self._logger.warn("post write")
        return None


    def plot_points(self) -> None:
        nodes = [(0.0, 0.0, 0.0), (-15, -15, 0), (-5, -15, 0), (5, -15, 0), (15, -15, 0),
                 (-20, -10, 0), (-20, -5, 0), (-20, 5, 0), (-20, 0, 0), (-20, 10, 0), (-20, -15, 0)]
        for node in nodes:
            self._plot_node_on_map(node)


    def find_best_fit_function(self, x, y, degree=5):
        """
        Finds the best-fitting polynomial function for a given set of x and y values.

        Parameters:
        - x: list of floats (input values)
        - y: list of floats (output values)
        - degree: int (degree of the polynomial to fit)

        Returns:
        - poly: Polynomial object representing the best-fit function
        """
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        
        # Fit the polynomial of the given degree
        coeffs = np.polyfit(x, y, degree)
        
        # Create a polynomial function from the coefficients
        poly = np.poly1d(coeffs)
        
        return poly
    

    def generate_map(self) -> None:
        """
        Generates final path image.
        """
        map_folder: str = os.path.join("images/maps", self._map_name)
        file_name: str = "final_" + self._trajectory + ".png"
        output_path: str = os.path.join(f"images/adjusted_maps/{self._map_name[:-4]}", file_name)

        map_img = cv2.imread(map_folder)

        for _, pose, _ in self._images_pose:
            y, x, _ = pose
            px, py = self.world_to_pixel(-x, y, map_img.shape, self._world_limits, origin=self._origin)
            cv2.circle(map_img, (px, py), 1, (255, 0, 0), -1)

        for node, _ in self.graph:
            y, x, _ = node.pose
            px, py = self.world_to_pixel(-x, y, map_img.shape, self._world_limits, origin=self._origin)
            cv2.circle(map_img, (px, py), 5, (0, 0, 255), -1)

        cv2.imwrite(output_path, map_img)

        return None
    

    def generate_map_edges(self) -> None:
        """
        Generates final path image, drawing both node positions and edges.
        """
        map_folder: str = os.path.join("images/maps", self._map_name)
        file_name: str = "final_" + self._trajectory + ".png"
        output_path: str = os.path.join("images/final_edges_maps", file_name)

        map_img = cv2.imread(map_folder)

        # Draw nodes
        for node, _ in self.graph:
            y, x, _ = node.pose  # Assuming pose = (y, x, theta)
            px, py = world_to_pixel(-x, y, map_img.shape, self._world_limits, origin=self._origin)
            cv2.circle(map_img, (px, py), 5, (0, 0, 255), -1)

        for node_1, node_2 in self.graph:
            if node_1 is not None and node_2 is not None:
                y1, x1, _ = node_1.pose
                y2, x2, _ = node_2.pose
                p1 = world_to_pixel(-x1, y1, map_img.shape, self._world_limits, origin=self._origin)
                p2 = world_to_pixel(-x2, y2, map_img.shape, self._world_limits, origin=self._origin)
                cv2.line(map_img, p1, p2, (0, 255, 0), 2)  # Green lines for edges

        cv2.imwrite(output_path, map_img)

        return None


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
            return self.concat_images(image_1, image_2)  # Low match count, maybe no overlap

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

    
    def world_to_pixel(self, x, y, map_shape, world_limits, origin) -> tuple[int, int]:
        """
        Converts world coordinates (x, y) to pixel coordinates in the image.
        Ensures (0,0) correctly maps to the designated cross in the map.

        Args:
            x (_type_): _description_
            y (_type_): _description_
            map_shape (_type_): _description_
            world_limits (_type_): _description_
            origin_px (_type_): _description_

        Returns:
            _type_: _description_
        """
        x_min, x_max, y_min, y_max = world_limits
        map_h, map_w = map_shape[:2]
        origin_x, origin_y = origin
        
        scale_x = map_w / (x_max - x_min)
        scale_y = map_h / (y_max - y_min)
        
        px = int(origin_x + (x * scale_x))
        py = int(origin_y - (y * scale_y))

        return px, py
    

    def concat_images(self, image_1: np.ndarray, image_2: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        Concatenates two images side by side (horizontally) or on top of each other (vertically).

        Args:
            img1 (np.ndarray): first image.
            img2 (np.ndarray): second image.
            axis (int): 0 for vertical stack, 1 for horizontal stack (default).

        Returns:
            np.ndarray: concatenated image.
        """
        if axis == 1 and image_1.shape[0] != image_2.shape[0]:  # Horizontal
            h: int = min(image_1.shape[0], image_2.shape[0])
            image_1 = cv2.resize(image_1, (int(image_1.shape[1] * h / image_1.shape[0]), h))
            image_2 = cv2.resize(image_2, (int(image_2.shape[1] * h / image_2.shape[0]), h))
        elif axis == 0 and image_1.shape[1] != image_2.shape[1]:  # Vertical
            w: int = min(image_1.shape[1], image_2.shape[1])
            image_1 = cv2.resize(image_1, (w, int(image_1.shape[0] * w / image_1.shape[1])))
            image_2 = cv2.resize(image_2, (w, int(image_2.shape[0] * w / image_2.shape[1])))

        return np.concatenate((image_1, image_2), axis=axis)


def crop_black_borders(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
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


def concat_images(image_1: np.ndarray, image_2: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Concatenates two images side by side (horizontally) or on top of each other (vertically).

    Args:
        img1 (np.ndarray): first image.
        img2 (np.ndarray): second image.
        axis (int): 0 for vertical stack, 1 for horizontal stack (default).

    Returns:
        np.ndarray: concatenated image.
    """
    if axis == 1 and image_1.shape[0] != image_2.shape[0]:  # Horizontal
        h: int = min(image_1.shape[0], image_2.shape[0])
        image_1 = cv2.resize(image_1, (int(image_1.shape[1] * h / image_1.shape[0]), h))
        image_2 = cv2.resize(image_2, (int(image_2.shape[1] * h / image_2.shape[0]), h))
    elif axis == 0 and image_1.shape[1] != image_2.shape[1]:  # Vertical
        w: int = min(image_1.shape[1], image_2.shape[1])
        image_1 = cv2.resize(image_1, (w, int(image_1.shape[0] * w / image_1.shape[1])))
        image_2 = cv2.resize(image_2, (w, int(image_2.shape[0] * w / image_2.shape[1])))

    return np.concatenate((image_1, image_2), axis=axis)


def process_stitched_image(image: np.ndarray) -> torch.Tensor:
    """
    Converts an OpenCV image (np.ndarray) to a PyTorch tensor after applying transformations.

    Args:
        image (np.ndarray): the stitched image in BGR format.

    Returns:
        torch.Tensor: transformed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Convert from BGR (OpenCV) to RGB (PIL expects RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert NumPy array to PIL Image
    image_pil = Image.fromarray(image_rgb)

    # Apply transformations
    image_tensor = transform(image_pil)

    return image_tensor


def world_to_pixel(x, y, map_shape, world_limits, origin) -> tuple[int, int]:
    """
    Converts world coordinates (x, y) to pixel coordinates in the image.
    Ensures (0,0) correctly maps to the designated cross in the map.

    Args:
        x (_type_): _description_
        y (_type_): _description_
        map_shape (_type_): _description_
        world_limits (_type_): _description_
        origin_px (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_min, x_max, y_min, y_max = world_limits
    map_h, map_w = map_shape[:2]
    origin_x, origin_y = origin
    
    scale_x = map_w / (x_max - x_min)
    scale_y = map_h / (y_max - y_min)
    
    px = int(origin_x + (x * scale_x))
    py = int(origin_y - (y * scale_y))

    return px, py