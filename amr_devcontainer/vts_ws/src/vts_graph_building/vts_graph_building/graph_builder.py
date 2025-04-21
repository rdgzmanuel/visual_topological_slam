import numpy as np
import cv2
import os
import rclpy.logging
import torch
import time
from torchvision import transforms
import gtsam
from gtsam import Pose2, BetweenFactorPose2, noiseModel
from PIL import Image
from vts_graph_building.node import GraphNodeClass
from vts_camera.camera import Camera


class GraphBuilder:
    def __init__(self, n: int, gamma_proportion: float, delta_proportion: float, distance_threshold: float,
                 initial_pose: tuple[float, float, float], world_limits: tuple[float, float, float, float],
                 map_name: str, origin: tuple[int, int], weights: tuple[float], trajectory: str,
                 model_name: str):
        """
        Constructor of the GraphBuilder class. Contains relevant graph information
        and hyperparameters.

        Args:
            initial_pose (tuple[float, float, float], optional): initial pose of the
            system. Defaults to (0.0, 0.0, 0.0).
        """
        self._initial_pose: tuple[float, float, float] = initial_pose
        self.current_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.current_node: GraphNodeClass | None = None
        self._current_image: np.ndarray | None = None
        self.steps: int = 0
        self.graph: list[tuple[GraphNodeClass, GraphNodeClass]] = []
        self.window_full: bool = False

        self._images_pose: list[tuple[np.ndarray, tuple[float, float, float], np.ndarray]] = []
        self.window_images: np.ndarray | None = None
        self._eigenvalues: list[float] = []
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

        # images stitching
        self._min_matches: int = 5
        self._min_descriptors: int = 2
        self._camera: Camera = Camera(model_name)
        self._image_shape: tuple[int, int, int] | None = None

        # rewiring
        self._rewiring_threshold: float = 3.0
        self._external_rewiring_threshold: float = 1.5

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


    def new_upgrade_pose(self, image_name: str) -> None:
        """
        Upgrades system's pose using directly images names and performing a transformaton.

        Args:
            image_name (str): _description_
        """
        self._current_image: np.ndarray = self._convert_image(image_name)
        # self._logger.warn("post convrt")

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
            x (float): current x coordinate.
            y (float): current y coordinate.
            prev_x (float): previous x coordinate.
            prev_y (float): previous y coordinate.

        Returns:
            float: distance between (x, y) and (prev_x, prev_y).
        """
        return np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)


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
        self._eigenvalues.append(lambda_2)

        return lambda_2


    def look_for_valley(self) -> int:
        """
        Applies peak and valley finding algorithm.

        Returns:
            bool: _description_
        """
        current_image_idx: int = len(self._images_pose) - 1

        lambda_2: float = self._obtain_eigenvalue()

        valley_idx: int = 0

        if lambda_2 > self.max_value:
            self.max_value = lambda_2
        elif lambda_2 < self.min_value:
            self.min_value = lambda_2
            self.min_idx = current_image_idx
        
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
            self.current_node = new_node
            self._plot_node_on_map(self.current_node.pose)

        else:
            closest_neighbor: GraphNodeClass = self._search_closest_neighbor((new_node.pose[0], new_node.pose[1]))
            distance: float = self._compute_distance(closest_neighbor.pose[0], closest_neighbor.pose[1],
                                                     new_node.pose[0], new_node.pose[1])

            if distance < self._distance_threshold:
                # Revisit confirmed
                closest_neighbor = self._fusion_nodes(new_node, closest_neighbor)
                loop_nodes: list[GraphNodeClass] = self._obtain_loop_nodes(closest_neighbor)
                # self._logger.warn(f"Loop nodes {[node.id for node in loop_nodes]}")
                relevant_edges: list[tuple[GraphNodeClass, GraphNodeClass]] = self._obtain_relevant_edges(loop_nodes)
                self._optimize_loop_poses(loop_nodes, relevant_edges)
                # self._logger.warn(f"Opitimizing time {time.time() - start}")
                self._rewire_graph(loop_nodes, relevant_edges, self._rewiring_threshold)
                self.current_node = closest_neighbor
                # self._logger.warn(f"Finsished rewiring {[(node.id, adj.id) for node, adj in self.graph]}")
                # self._logger.warn(f"Poses {[(node.id, node.pose) for node, _ in self.graph]}")

            else:
                self._node_id += 1
                                   
                # self._logger.warn(f"add {self.current_node.pose[:2]} {new_node.pose[:2]}")
                self.graph.append((self.current_node, new_node))

                # Before adding a new edge, check current node's neighbors and edges to see if we can project them
                relevant_edges: list[tuple[GraphNodeClass, GraphNodeClass]] = [edge for edge in self.graph if self.current_node in edge] 
                self._rewire_graph([new_node], relevant_edges, self._external_rewiring_threshold)

                relevant_nodes: list[GraphNodeClass] = list(self.current_node.neighbors)
                self._rewire_graph(relevant_nodes, [(self.current_node, new_node)], self._external_rewiring_threshold)

                new_node.neighbors.add(self.current_node)
                self.current_node.neighbors.add(new_node)
                self.current_node = new_node
                self._plot_node_on_map(self.current_node.pose)
                # self._logger.warn("finish update")

        return None
    

    def _search_closest_neighbor(self, pose: tuple[float, float]) -> GraphNodeClass:
        """
        

        Args:
            pose (tuple[float, float]): _description_

        Returns:
            GraphNodeClass: _description_
        """
        closest_neighbor: GraphNodeClass = self.current_node
        min_distance: float = self._compute_distance(pose[0], pose[1],
                                                     closest_neighbor.pose[0], closest_neighbor.pose[1])

        for node, _ in self.graph:
            distance: float = self._compute_distance(pose[0], pose[1], node.pose[0], node.pose[1])
            if distance < min_distance:
                min_distance = distance
                closest_neighbor = node
        
        return closest_neighbor
    

    def _fusion_nodes(self, new_node: GraphNodeClass, closest_neighbor: GraphNodeClass) -> GraphNodeClass:
        """
        Fusions new node with its closests neighbor, stitching their images and feeding the
        result to the model to obtain new features.

        Args:
            new_node (GraphNodeClass): _description_
            closets_neighbor (GraphNodeClass): _description_
        """
        # new_pose = self._average_pose(closest_neighbor.pose, new_node.pose)
        new_image: np.ndarray = self.stitch_images(new_node.image, closest_neighbor.image,
                                            min_matches=self._min_matches)
        new_image = cv2.resize(new_image, (self._image_shape[1], self._image_shape[0]))
        tensor_image: torch.Tensor = process_stitched_image(new_image)
        new_visual_features: np.ndarray = self._extract_features(tensor_image)

        # closest_neighbor.pose = new_pose
        closest_neighbor.image = new_image
        closest_neighbor.visual_features = new_visual_features
        self._plot_node_on_map(closest_neighbor.pose)

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
        return tuple((a + b) / 2 for a, b in zip(pose_1, pose_2))


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
    

    def _obtain_loop_nodes(self, closest_neighbor: GraphNodeClass) -> list[GraphNodeClass]:
        """
        Returns all nodes between current node and closest neighbor (nodes in the loop).

        Args:
            closest_neighbor (GraphNodeClass): _description_

        Returns:
            list[GraphNodeClass]: _description_
        """
        loop_nodes: list[GraphNodeClass] = []
        # node: GraphNodeClass = self.graph[-1][0]
        node: GraphNodeClass = self.current_node

        if node != closest_neighbor:
            i: int = 2
            while node != closest_neighbor:
                loop_nodes.append(node)
                node = self.graph[-i][0]
                i += 1
            
            loop_nodes.append(closest_neighbor)

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

        for node in loop_nodes:
            x, y, _ = node.pose
            projections: dict[tuple[GraphNodeClass, GraphNodeClass], tuple[float, float]] = self._get_projections(node, relevant_edges)
            for edge, projection in projections.items():
                if projection is not None:
                    distance: float = self._compute_distance(x, y, projection[0], projection[1])

                    if distance < threshold:
                        # self._logger.warn(f"Add {edge[0].pose[:2]} {node.pose[:2]} {edge[1].pose[:2]}")
                        # rewired edge found
                        self.graph.append((edge[0], node))
                        self.graph.append((node, edge[1]))
                        edge[0].neighbors.add(node)
                        edge[1].neighbors.add(node)
                        node.neighbors.add(edge[0])
                        node.neighbors.add(edge[1])
                        if edge in self.graph:
                            # self._logger.warn(f"Removes {edge[0].pose[:2]} {edge[1].pose[:2]}")
                            self.graph.remove(edge)
                            if (edge[1], edge[0]) in self.graph:
                                self.graph.remove((edge[1], edge[0]))
                                if edge[1] in edge[0].neighbors:
                                    edge[0].neighbors.remove(edge[1])
                                if edge[0] in edge[1].neighbors:
                                    edge[1].neighbors.remove(edge[0])
                                


    def _get_projections(self, node: GraphNodeClass,
                         relevant_edges: list[tuple[GraphNodeClass, GraphNodeClass]]) -> dict[tuple[GraphNodeClass, GraphNodeClass],
                                                                                              tuple[float, float]]:
        """
        

        Args:
            node (GraphNodeClass): _description_
            relevant_edges (list[tuple[GraphNodeClass, GraphNodeClass]]): _description_

        Returns:
            dict[tuple[GraphNodeClass, GraphNodeClass], tuple[float, float]]: _description_
        """
        projections: dict[tuple[GraphNodeClass, GraphNodeClass], tuple[float, float]] = {}

        for edge in relevant_edges:
            if node not in edge:
                projection: tuple[float, float] = self._project_node(node, edge)
                edge_distance: float = self._compute_distance(edge[0].pose[0], edge[0].pose[1], edge[1].pose[0], edge[1].pose[1])

                x: float
                y: float
                x, y = projection
                first_distance: float = self._compute_distance(edge[0].pose[0], edge[0].pose[1], x, y)
                second_distance: float = self._compute_distance(edge[1].pose[0], edge[1].pose[1], x, y)

                if np.isclose(edge_distance, (first_distance + second_distance), rtol=1e-3, atol=1e-3):
                    projections[edge] = projection
        
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
        x1, y1, _ = edge[0].pose  # Coordinates of the first node in the edge
        x2, y2, _ = edge[1].pose  # Coordinates of the second node in the edge
        x, y, _ = node.pose       # Coordinates of the node to project

        v_x, v_y = x2 - x1, y2 - y1  # Edge vector (v)
        u_x, u_y = x - x1, y - y1    # Vector from first node to the node (u)

        # Compute the dot product of u and v
        dot_product: float = u_x * v_x + u_y * v_y
        
        # Compute the squared magnitude of the edge vector (v)
        v_squared_magnitude: float = v_x ** 2 + v_y ** 2
        
        # Compute the projection scalar
        projection_scalar: float = dot_product / v_squared_magnitude
        
        # Compute the projected point on the edge
        proj_x: float = x1 + projection_scalar * v_x
        proj_y: float = y1 + projection_scalar * v_y
        
        return proj_x, proj_y
    

    def _optimize_loop_poses(
        self,
        nodes: list[GraphNodeClass],
        graph_edges: list[tuple[GraphNodeClass, GraphNodeClass]]
    ) -> None:
        """
        Applies pose graph optimization to correct loop closure drift.
        The first node in `nodes` is assumed to be the fixed reference (e.g., corridor entry).

        Args:
            nodes (List[GraphNodeClass]): All nodes in the loop, including the fixed corridor node.
            graph_edges (List[Tuple[GraphNodeClass, GraphNodeClass]]): List of relative pose constraints between nodes.
        """
        graph: gtsam.NonlinearFactorGraph = gtsam.NonlinearFactorGraph()
        initial_estimates: gtsam.Values = gtsam.Values()

        # --- Step 1: Add a prior to fix the first node (corridor node) ---
        if len(nodes) > 0:
            fixed_node: GraphNodeClass = nodes[0]
            fixed_pose: Pose2 = Pose2(*fixed_node.pose)
            prior_noise = noiseModel.Diagonal.Sigmas([1e-6, 1e-6, 1e-6])
            graph.add(gtsam.PriorFactorPose2(fixed_node.id, fixed_pose, prior_noise))
            initial_estimates.insert(fixed_node.id, fixed_pose)

            # --- Step 2: Add binary relative pose constraints (edges) ---
            for node1, node2 in graph_edges:
                pose1: Pose2 = Pose2(*node1.pose)
                pose2: Pose2 = Pose2(*node2.pose)
                relative_pose: Pose2 = pose1.between(pose2)
                model = noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1])
                graph.add(BetweenFactorPose2(node1.id, node2.id, relative_pose, model))

                # Insert initial guesses if not already present
                if not initial_estimates.exists(node1.id):
                    initial_estimates.insert(node1.id, pose1)
                if not initial_estimates.exists(node2.id):
                    initial_estimates.insert(node2.id, pose2)

            # --- Step 3: Optimize the pose graph ---
            params: gtsam.LevenbergMarquardtParams = gtsam.LevenbergMarquardtParams()
            optimizer: gtsam.LevenbergMarquardtOptimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
            result: gtsam.Values = optimizer.optimize()

            # --- Step 4: Update the node poses with the optimized result ---
            for node in nodes:
                if result.exists(node.id):
                    optimized_pose: Pose2 = result.atPose2(node.id)
                    node.pose = (optimized_pose.x(), optimized_pose.y(), optimized_pose.theta())
        
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

        x, y, _ = pose  # Ignore theta for now
        px, py = self.world_to_pixel(x, y, map_img.shape, self._world_limits, self._origin)
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
            x, y, _ = pose
            px, py = self.world_to_pixel(x, y, map_img.shape, self._world_limits, origin=self._origin)
            cv2.circle(map_img, (px, py), 1, (255, 0, 0), -1)

        for node, _ in self.graph:
            x, y, _ = node.pose
            px, py = self.world_to_pixel(x, y, map_img.shape, self._world_limits, origin=self._origin)
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
            x, y, _ = node.pose  # Assuming pose = (y, x, theta)
            px, py = world_to_pixel(x, y, map_img.shape, self._world_limits, origin=self._origin)
            cv2.circle(map_img, (px, py), 5, (0, 0, 255), -1)

        # Draw edges
        for node_1, node_2 in self.graph:
            if node_1 is not None and node_2 is not None:
                x1, y1, _ = node_1.pose
                x2, y2, _ = node_2.pose
                p1 = world_to_pixel(x1, y1, map_img.shape, self._world_limits, origin=self._origin)
                p2 = world_to_pixel(x2, y2, map_img.shape, self._world_limits, origin=self._origin)
                cv2.line(map_img, p1, p2, (0, 255, 0), 2)  # Green lines for edges

        cv2.imwrite(output_path, map_img)

        return None
    

    def stitch_images(self, image_1: np.ndarray, image_2: np.ndarray, min_matches: int = 5) -> np.ndarray:
        """
        Stitches two images together using feature matching and homography.

        Args:
            img1 (str): first input image path.
            img2 (str): second input image (BGR format).

        Returns:
            np.ndarray: stitched image if successful, otherwise concated images.
        """

        gray_1: np.ndarray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        gray_2: np.ndarray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp1: list[cv2.KeyPoint]
        kp2: list[cv2.KeyPoint]
        des1: np.ndarray
        des2: np.ndarray
        kp1, des1 = sift.detectAndCompute(gray_1, None)
        kp2, des2 = sift.detectAndCompute(gray_2, None)

        # self._logger.warn("pre flann")

        # FLANN based feature matching
        index_params: dict[str, int] = {"algorithm": 1, "trees": 5}
        search_params: dict[str, int] = {"checks": 50}
        flann: cv2.FlannBasedMatcher = cv2.FlannBasedMatcher(index_params, search_params)


        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            stitched_image: np.ndarray = self.concat_images(image_1, image_2)
        else:
            matches: list[list[cv2.DMatch]] = flann.knnMatch(des1, des2, k=self._min_descriptors)

            # Lowe's ratio test to keep good matches
            good_matches: list[cv2.DMatch] = [
                m for m, n in matches if m.distance < 0.7 * n.distance
            ]

            if len(good_matches) > min_matches:
                src_pts: np.ndarray = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts: np.ndarray = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Compute homography
                H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                # Warp first image to align with the second
                height, width, _ = image_2.shape
                if H is not None and H.shape == (3, 3):
                    H = H.astype(np.float32)  # or np.float64
                    warped_img1 = cv2.warpPerspective(image_1, H, (width * 2, height))
                    # Place second image on the stitched result
                    warped_img1[0:height, 0:width] = image_2

                    # Convert to grayscale and find non-zero regions for blending
                    gray_warped: np.ndarray = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)
                    # self._logger.warn("pre block")
                    # Crop the stitched image
                    stitched_image = self.crop_black_borders(warped_img1, mask)
                else:
                    stitched_image = self.concat_images(image_1, image_2)
            
            else:
                stitched_image = self.concat_images(image_1, image_2)

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


def stitch_images(image_1: np.ndarray, image_2: np.ndarray, min_matches: int = 10) -> np.ndarray:
    """
    Stitches two images together using feature matching and homography.

    Args:
        img1 (str): first input image path.
        img2 (str): second input image (BGR format).

    Returns:
        np.ndarray: stitched image if successful, otherwise concated images.
    """

    gray_1: np.ndarray = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    gray_2: np.ndarray = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1: list[cv2.KeyPoint]
    kp2: list[cv2.KeyPoint]
    des1: np.ndarray
    des2: np.ndarray
    kp1, des1 = sift.detectAndCompute(gray_1, None)
    kp2, des2 = sift.detectAndCompute(gray_2, None)

    if des1 is None or des2 is None:
        raise ValueError("Descriptors could not be computed. One of the images has too few features.")


    # FLANN based feature matching
    index_params: dict[str, int] = {"algorithm": 1, "trees": 5}
    search_params: dict[str, int] = {"checks": 50}
    flann: cv2.FlannBasedMatcher = cv2.FlannBasedMatcher(index_params, search_params)

    matches: list[list[cv2.DMatch]] = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test to keep good matches
    good_matches: list[cv2.DMatch] = [
        m for m, n in matches if m.distance < 0.7 * n.distance
    ]

    if len(good_matches) > min_matches:
        src_pts: np.ndarray = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts: np.ndarray = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute homography
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            raise ValueError("cv2.findHomography failed to compute a valid matrix.")

        # Warp first image to align with the second
        height, width, _ = image_2.shape
        if H is not None and H.shape == (3, 3):
            H = H.astype(np.float32)  # or np.float64
            warped_img1 = cv2.warpPerspective(image_1, H, (width * 2, height))
        else:
            raise ValueError(f"Homography matrix H is invalid. Check shape and computation. {H.shape} {H.dtype}")

        # Place second image on the stitched result
        warped_img1[0:height, 0:width] = image_2

        # Convert to grayscale and find non-zero regions for blending
        gray_warped: np.ndarray = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_warped, 1, 255, cv2.THRESH_BINARY)

        # Crop the stitched image
        stitched_image: np.ndarray = crop_black_borders(warped_img1, mask)
    
    else:
        stitched_image = concat_images(image_1, image_2)

    return stitched_image


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