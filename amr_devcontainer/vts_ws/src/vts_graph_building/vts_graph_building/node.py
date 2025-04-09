import numpy as np


class GraphNode:
    def __init__(self, id: int, pose: tuple[float, float, float], visual_features: np.ndarray, image: np.ndarray):
        self.pose: tuple[float, float, float] = pose
        self.visual_features: np.ndarray = visual_features
        self.neighbors: list[GraphNode] = []
        self.id: int = id
        self.image: np.ndarray = image