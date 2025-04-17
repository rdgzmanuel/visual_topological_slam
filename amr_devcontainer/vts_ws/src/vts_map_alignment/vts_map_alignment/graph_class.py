from vts_graph_building.node import GraphNodeClass


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[int, GraphNodeClass] = {}
        self.edges: list[tuple[int]] = []
        self.current_node: GraphNodeClass | None = None
        self.node_id: int = 0
