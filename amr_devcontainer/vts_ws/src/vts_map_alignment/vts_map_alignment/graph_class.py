from vts_graph_building.node import GraphNode


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[int, GraphNode] = {}
        self.edges: list[tuple[int]] = []
        self.current_node: GraphNode | None = None
        self.node_id: int = 0
