import numpy as np
import torch
import faiss
import torch.nn.functional as F

from node import GraphNode


class GraphBuilder:
    def __init__(self, n: int, gamma_proportion: float, delta_proportion: float,
                 initial_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)):
        """
        Constructor of the GraphBuilder class. Contains relevant graph information
        and hyperparameters.

        Args:
            initial_pose (tuple[float, float, float], optional): initial pose of the
            system. Defaults to (0.0, 0.0, 0.0).
        """
        self._initial_pose: tuple[float, float, float] = initial_pose
        self.current_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._current_node: GraphNode | None = None
        self.steps: int = 0
        self.graph: list[tuple[GraphNode, GraphNode]] = []
        self.window_full: bool = False

        self._images_pose: list[tuple[np.ndarray, tuple[float, float, float]]] = []
        self._window_images: list[tuple[np.ndarray, int]] = []
        self._eigenvalues: list[float] = []
        self._representative_candidates: list[tuple[int, float]] = []


        self._n: int = n
        self._lambda_2_max: float = n / (n - 1)
        self._gamma: float = gamma_proportion * self._lambda_2_max
        self._delta: float = delta_proportion * self._gamma

        self._max_similarity: float = 0
        self._max_index: int = 0
        
        self._current_alg_conenctivity: float = 0.0
        self._node_id = 0
    
    def update_pose(self, v: float, w: float, time_difference: float) -> None:
        """
        Updates current pose of the system with osometry measurements

        Args:
            v (float): lineal velocity
            w (float): angular velocity
            time_difference (float): time difference between previous and current pose
        """

        prev_x: float
        prev_y: float
        prev_theta: float

        if self.steps == 0:
            prev_x, prev_y, prev_theta = self._initial_pose
        else:
            prev_x, prev_y, prev_theta = self.current_pose

        x: float = prev_x + v * np.cos(prev_theta + w * time_difference / 2) * time_difference
        y: float = prev_y + v * np.sin(prev_theta + w * time_difference / 2) * time_difference
        theta: float = (prev_theta + w * time_difference) % (2 * np.pi)

        self.current_pose = (x, y, theta)
        self.steps += 1

        return None


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
        self._window_images.pop(0)

        return None


    def _compute_similarities(self, array_data: np.ndarray) -> None:
        """
        Updates affinity matrix computing the similarities with previous images from the
        window and current incoming image.

        Args:
            idx (int): _description_
        """

        similarities: np.ndarray = np.dot(self._window_images, array_data)

        self.affinity[:-1, -1] = similarities
        self.affinity[-1, :-1] = similarities
        self.affinity[-1, -1] = 0.0

        return None
    

    def _update_degree_matrix(self) -> None:
        """
        Updates degree matrix with the new affinity matrix. Also looks for a new maximum
        in the similarity score.
        """
        self.degree: np.ndarray = np.diag(self.affinity.sum(axis=1))
        max_idx: int = np.argmax(self.degree)

        if self.degree[max_idx] > self._max_similarity:
            self._max_similarity = self.degree[max_idx]
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
        self._images_pose.append((norm_array_data, self.current_pose))
        self._window_images.append((norm_array_data), len(self._images_pose - 1))

        if len(self._window_images) <= self._n:
            # still building affinity matrix
            self._build_affinity_matrix()
        else:
            # affinity matrix already built, need to move window
            self._move_window()

        self._compute_similarities(norm_array_data)
        self._update_degree_matrix()
        self._update_laplacian_sym_matrix()

        self.window_full = len(self._window_images) == self._n

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

        return eigenvalues[1]


    def look_for_valley(self) -> tuple[bool, int]:
        """
        

        Returns:
            bool: _description_
        """
        current_image_idx: int = len(self._images_pose) - 1
        look_for_maximum: bool = True
        max_value: float = float("-inf")
        min_value: float = float("inf")
        min_idx: int = 0

        lambda_2: float = self._obtain_eigenvalue()

        valley_idx: int = 0
        found: bool = False

        # Check whether current algebraic connectivity is an extremum
        if lambda_2 > max_value:
            max_value = lambda_2
        elif lambda_2 < min_value:
            min_value = lambda_2
            min_idx = current_image_idx
        
        # Look for peaks and valleys
        if look_for_maximum:
            if self._current_alg_conenctivity < (max_value - self._delta) and max_value >= self._gamma:
                look_for_maximum = False
                min_value = lambda_2
        else:
            if self._current_alg_conenctivity > min_value + self._delta:
                look_for_maximum = True
                max_value = lambda_2
                valley_idx = min_idx
                found = True
        
        self._current_alg_conenctivity = lambda_2
        
        return found, valley_idx


    def update_graph(self, valley_idx: int) -> None:
        """
        Creates a new node with selected representative and updates relevant variables.

        Args:
            valley_idx (int): _description_
        """
        representative: tuple[np.ndarray, tuple[float, float, float]] = self._images_pose[self._max_index]

        self._max_similarity = float("-inf")

        new_node: GraphNode = GraphNode(id=self._node_id, pose=representative[1],
                                        visual_features=representative[0])
        self._node_id += 1

        if self._current_node is not None:
            self.graph.append((self._current_node, new_node))
        
        self._current_node = new_node

        cut_index: int = 0
        while self._window_images[cut_index][1] < valley_idx:
            cut_index += 1
        
        self.affinity = self.affinity[cut_index:, cut_index:]
        self._window_images = self._window_images[cut_index:]


