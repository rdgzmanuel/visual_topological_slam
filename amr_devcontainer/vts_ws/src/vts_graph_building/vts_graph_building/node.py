class Node:
    def __init__(self, id: int, position: tuple[float, float], visual_features: list[float]):
        self.position: tuple[float, float] = position
        self.visual_features: list[float] = visual_features
        self.neighbors: list[Node] = []
        self.id: int = id