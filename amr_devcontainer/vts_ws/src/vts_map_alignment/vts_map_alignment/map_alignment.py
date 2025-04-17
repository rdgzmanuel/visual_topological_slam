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
from vts_graph_building.graph_builder import stitch_images, crop_black_borders, concat_images,\
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
        self._pose_weight: float = 0.75
        self._threshold: float = 2.0

        # Image stitching
        self._min_matches: int = 10
        self._camera: Camera = Camera(model_name)

        self._logger = rclpy.logging.get_logger('MapAlignment')


    def align_graphs(self, graph_1: Graph, graph_2: Graph) -> None:
        """


        Args:
            graph_1 (Graph): _description_
            graph_2 (Graph): _description_

        Returns:
            Graph: _description_
        """
        self.updated_graph: Graph = copy.deepcopy(graph_2)
        # for node in graph_1.nodes.values():
        #     matches: list[GraphNodeClass] = self._update_graph(node, graph_2)

        # for node in graph_2.nodes.values():
        #     if node not in matches:
        #         _ = self._update_graph(node, graph_1, update_matches=False)
        
        return None


    def _update_graph(self, node: GraphNodeClass, lookup_graph: Graph,
                       update_matches: bool = True) -> list[GraphNodeClass]:
        """
        

        Args:
            node (GraphNodeClass): _description_
            updated_graph (Graph): _description_
            lookup_graph (Graph): _description_
            update_matches (bool, optional): _description_. Defaults to True.
            matches (list[GraphNodeClass], optional): _description_. Defaults to [].
        """
        matches: list[GraphNodeClass] = []

        best_match: GraphNodeClass = self._search_best_match(node, lookup_graph)
        new_id: int = self.updated_graph.node_id
        self.updated_graph.node_id += 1

        if best_match is not None:
            new_node = self._fusion_nodes(node, best_match, new_id)
            if update_matches:
                matches.append(best_match)
        else:
            new_node = copy.deepcopy(node)
            new_node.id = new_id

        if self.updated_graph.current_node is not None:
            self.updated_graph.edges.append((self.updated_graph.current_node.id, new_node.id))
        
        self.updated_graph.nodes[new_node.id] = new_node
        self.updated_graph.current_node = new_node

        return matches
    

    def _search_best_match(self, node: GraphNodeClass, lookup_graph: Graph, k: int = 3) -> GraphNodeClass:
        """
        Find the best matching node to the given node in the lookup graph using a KD-tree and cosine similarity.

        Args:
            node (GraphNodeClass): The query node.
            lookup_graph (Graph): The graph containing nodes to search.
            k (int): The number of nearest neighbors to find.

        Returns:
            GraphNodeClass: The best matching node.
        """
        node_positions: np.ndarray = np.array([n.pose[:2] for n in lookup_graph.nodes.values()])
        node_list: list[GraphNodeClass] = list(lookup_graph.nodes.values())

        kd_tree: KDTree = KDTree(node_positions)

        distances: np.ndarray
        indices: np.ndarray
        distances, indices = kd_tree.query(node.pose[:2], k=min(k, len(node_list)))
        
        candidates: list[GraphNodeClass] = [node_list[i] for i in np.atleast_1d(indices)]
        
        best_node: GraphNodeClass | None = None
        best_score: float = self._threshold

        for i, candidate in enumerate(candidates):
            pose_dist: float = distances[i]
            visual_sim: float = cosine(node.visual_features, candidate.visual_features)
            
            score: float = self._pose_weight * pose_dist + (1 - self._pose_weight) * visual_sim

            if score < best_score:
                best_score = score
                best_node = candidate

        return best_node


    def _fusion_nodes(self, node_1: GraphNodeClass, node_2: GraphNodeClass, new_id: int) -> GraphNodeClass:
        """
        

        Args:
            node_1 (GraphNodeClass): _description_
            node_2 (GraphNodeClass): _description_

        Returns:
            GraphNodeClass: _description_
        """

        new_pose: tuple[float, float, float] = self._average_pose(node_1.pose, node_2.pose)
        new_image: np.ndarray = stitch_images(node_1.image, node_2.image,
                                                min_matches=self._min_matches)
        tensor_image: torch.Tensor = process_stitched_image(new_image)
        new_visual_features: np.ndarray = self._extract_features(tensor_image)

        new_node: GraphNodeClass = GraphNodeClass(id=new_id, pose=new_pose, visual_features=new_visual_features,
                                        image=new_image)

        return new_node
    

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
        file_name: str = "final_" + self._trajectory + ".png"
        output_path: str = os.path.join("images/final_aligned_maps", file_name)

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