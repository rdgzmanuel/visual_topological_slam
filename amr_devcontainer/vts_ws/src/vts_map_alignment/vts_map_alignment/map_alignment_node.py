import numpy as np
import rclpy
import sys
from rclpy.node import Node
from collections import deque
from typing import Deque
from vts_msgs.msg import FullGraph
from vts_graph_building.node import GraphNode
from vts_map_alignment.graph_class import Graph
from vts_map_alignment.map_alignment import MapAligner


class GraphAlignment(Node):
    def __init__(self) -> None:
        super().__init__("graph_alignment")

        self.declare_parameter("model_name", "default_value")
        model_name: str = self.get_parameter("model_name").get_parameter_value().string_value

        self.declare_parameter("trajectory", "default_value")
        trajectory: str = self.get_parameter("trajectory").get_parameter_value().string_value

        self.declare_parameter("map_name", "default_value")
        map_name: str = self.get_parameter("map_name").get_parameter_value().string_value

        self.declare_parameter("origin", (0, 0))
        origin: tuple[int, int] = tuple(self.get_parameter("origin").
                                        get_parameter_value().integer_array_value.tolist())
        
        self.declare_parameter("world_limits", (0.0, 0.0, 0.0, 0.0))
        world_limits: tuple[float, float, float, float] = tuple(self.get_parameter("world_limits").
                                                                 get_parameter_value().double_array_value.tolist())

        self._graph_subscriber = self.create_subscription(
            FullGraph, "/graph_alignment", self.graph_message_callback, 10
        )
        
        self._graph_queue: Deque[FullGraph] = deque(maxlen=2)

        self._map_aligner: MapAligner = MapAligner(model_name, trajectory, world_limits, origin, map_name)


    def graph_message_callback(self, graph_msg: FullGraph) -> None:
        """
        Callback function to store messages and trigger processing when two messages are received.
        """
        self._graph_queue.append(graph_msg)
        self.get_logger().warn("Received a graph")
        
        if len(self._graph_queue) == 2:
            self.get_logger().warn("Received two graphs. Starting alignment...")
            graph_msg_1: FullGraph = self._graph_queue.popleft()
            graph_msg_2: FullGraph = self._graph_queue.popleft()
            self.graphs_callback(graph_msg_1, graph_msg_2)


    def graphs_callback(self, graph_msg_1: FullGraph, graph_msg_2: FullGraph) -> None:
        """
        Processes and aligns two received graphs.
        """
        self.get_logger().warn("Processing graphs")
        graph_1: Graph = self._process_graph_msg(graph_msg_1)
        graph_2: Graph = self._process_graph_msg(graph_msg_2)
        
        self.get_logger().warn("Aligning graphs")
        self._map_aligner.align_graphs(graph_1, graph_2)
        self.get_logger().warn("Generating maps")
        self._map_aligner.generate_map()
        self.get_logger().warn("Map generated. Shutting down node.")
        sys.exit(0)


    def _process_graph_msg(self, message: FullGraph) -> Graph:
        """
        Converts a FullGraph message into a Graph object.
        """
        graph: Graph = Graph()
        edges: list[int] = message.edges

        for node_message in message.nodes:
            image: np.ndarray = np.array(node_message.image).reshape(node_message.shape).astype(np.uint8)
            pose: tuple[float, float] = tuple(node_message.pose)
            features: np.ndarray = np.array(node_message.features)
            id: int = node_message.node_id

            new_node: GraphNode = GraphNode(id=id, pose=pose, visual_features=features, image=image)
            graph.nodes[id] = new_node

        for i in range(0, len(edges), 2):
            graph.edges.append((edges[i], edges[i + 1]))
        return graph


def main(args: list[str] = None) -> None:
    rclpy.init(args=args)
    graph_alignment_node: GraphAlignment = GraphAlignment()

    try:
        rclpy.spin(graph_alignment_node)
    except KeyboardInterrupt:
        pass

    graph_alignment_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
