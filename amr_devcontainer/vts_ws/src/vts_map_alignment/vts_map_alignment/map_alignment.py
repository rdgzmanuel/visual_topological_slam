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
        self._threshold: float = 1.5

        # Image stitching
        self._min_matches: int = 10
        self._camera: Camera = Camera(model_name)

        # Map alignment
        self._similarity_threshold: float = 0.65

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
        self._new_update_graph(graph_2)


    def _new_update_graph(self, lookup_graph: Graph) -> None:
        """
        Updates `self.updated_graph` with nodes and edges from a second graph.

        Args:
            lookup_graph (Graph): The graph whose nodes are being evaluated and fused/inserted.
        """
        for node in lookup_graph.nodes.values():
            best_match: GraphNodeClass | None = self._search_best_match(node)

            if best_match is not None:
                self._fusion_nodes(node, best_match)
                self.updated_graph.current_node = best_match
            else:
                
                new_node = copy.deepcopy(node)
                new_node.id = self.updated_graph.node_id

                self._include_node(new_node)

                self.updated_graph.nodes[new_node.id] = new_node
                self.updated_graph.current_node = new_node

                self.updated_graph.node_id += 1
        
        return None


    def _include_node(self, new_node: GraphNodeClass) -> None:
        """
        Adds a new node to the graph, connects it to the current node,
        and attempts to reroute an existing neighbor edge through this node.

        Args:
            new_node (GraphNodeClass): The new node to insert into the graph.
        """
        current_node = self.updated_graph.current_node
        next_node = self._find_next_node((current_node, new_node))
        self.updated_graph.edges.append((current_node.id, new_node.id))

        if next_node is not None:
            self.updated_graph.edges.append((new_node.id, next_node.id))
            # Remove direct connection from current_node to next_node if it exists
            
            direct_edge = (current_node.id, next_node.id)
            reverse_edge = (next_node.id, current_node.id)
            if direct_edge in self.updated_graph.edges:
                self.updated_graph.edges.remove(direct_edge)
            if reverse_edge in self.updated_graph.edges:
                self.updated_graph.edges.remove(reverse_edge)
        
        return None


    def _find_next_node(self, new_edge: tuple[GraphNodeClass, GraphNodeClass]) -> GraphNodeClass | None:
        """
        Finds the neighbor of the current node that most closely aligns
        (via dot product) with the direction of the new edge.

        Args:
            new_edge (tuple[GraphNodeClass, GraphNodeClass]): Tuple of (current_node, new_node)

        Returns:
            Optional[GraphNodeClass]: Neighbor node to connect through, or None if none match well.
        """
        current_node, new_node = new_edge
        direction_vector: np.array = np.array(new_node.pose[:2]) - np.array(current_node.pose[:2])

        max_similarity: float = float("-inf")
        best_match: None | GraphNodeClass = None

        relevant_edges_idx: list[tuple[int, int]] = [e for e in self.updated_graph.edges if current_node.id in e]
        relevant_edges: list[tuple[GraphNodeClass, GraphNodeClass]] = [(self.updated_graph.nodes[node_idx], self.updated_graph.nodes[adj_idx])
                                                                       for node_idx, adj_idx in relevant_edges_idx]

        for edge in relevant_edges:
            neighbor: GraphNodeClass = edge[1] if edge[0] == current_node else edge[0]
            edge_vector: np.array = np.array(neighbor.pose[:2]) - np.array(current_node.pose[:2])

            similarity: float = np.dot(direction_vector, edge_vector)

            if similarity > self._similarity_threshold and similarity > max_similarity:
                max_similarity = similarity
                best_match = neighbor

        return best_match


    def _search_best_match(self, node: GraphNodeClass, k: int = 3) -> GraphNodeClass:
        """
        Find the best matching node to the given node in the lookup graph using a KD-tree and cosine similarity.

        Args:
            node (GraphNodeClass): The query node.
            lookup_graph (Graph): The graph containing nodes to search.
            k (int): The number of nearest neighbors to find.

        Returns:
            GraphNodeClass: The best matching node.
        """
        node_positions: np.ndarray = np.array([n.pose[:2] for n in self.updated_graph.nodes.values()])
        node_list: list[GraphNodeClass] = list(self.updated_graph.nodes.values())

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


    def _fusion_nodes(self, node: GraphNodeClass, best_match: GraphNodeClass) -> None:
        """
        

        Args:
            node_1 (GraphNodeClass): Node coming from graph 2.
            node_2 (GraphNodeClass): Node coming from graph 1.

        Returns:
            GraphNodeClass: _description_
        """

        new_pose: tuple[float, float, float] = self._average_pose(node.pose, best_match.pose)
        new_image: np.ndarray = stitch_images(node.image, best_match.image,
                                                min_matches=self._min_matches)
        tensor_image: torch.Tensor = process_stitched_image(new_image)
        new_visual_features: np.ndarray = self._extract_features(tensor_image)

        best_match.pose = new_pose
        best_match.image = new_image
        best_match.visual_features = new_visual_features

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