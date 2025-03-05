class GraphNode:
    def __init__(self, id: int, pose: tuple[float, float, float], visual_features: list[float]):
        self.pose: tuple[float, float] = pose
        self.visual_features: list[float] = visual_features
        self.neighbors: list[GraphNode] = []
        self.id: int = id