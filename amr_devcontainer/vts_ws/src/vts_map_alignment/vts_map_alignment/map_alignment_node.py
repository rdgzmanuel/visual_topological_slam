import numpy as np
import rclpy
import os
import pickle
import sys
from rclpy.node import Node
from collections import deque
from typing import Deque
from vts_msgs.msg import FullGraph, CommandMessage
from vts_graph_building.node import GraphNodeClass
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
        self._map_name: str = self.get_parameter("map_name").get_parameter_value().string_value

        self.declare_parameter("origin", (0, 0))
        origin: tuple[int, int] = tuple(self.get_parameter("origin").
                                        get_parameter_value().integer_array_value.tolist())

        self.declare_parameter("world_limits", (0.0, 0.0, 0.0, 0.0))
        world_limits: tuple[float, float, float, float] = tuple(self.get_parameter("world_limits").
                                                                 get_parameter_value().double_array_value.tolist())

        self._graph_subscriber = self.create_subscription(
            FullGraph, "/graph_alignment", self.graph_message_callback, 10
        )

        self._graph_publisher = self.create_publisher(CommandMessage, "/commands", 10)

        self._graph_queue: Deque[FullGraph] = deque(maxlen=2)

        self._map_aligner: MapAligner = MapAligner(model_name, trajectory, world_limits, origin, self._map_name)

        # self._start_directly()


    def _start_directly(self) -> None:
        first_graph: str = "graph_1.pkl"
        second_graph: str = "graph_2.pkl"

        path: str = f"graphs/{self._map_name[:-4]}"

        graph_1: list[tuple[GraphNodeClass, GraphNodeClass]] = self.load_graph_data(os.path.join(path, first_graph))
        graph_2: list[tuple[GraphNodeClass, GraphNodeClass]] = self.load_graph_data(os.path.join(path, second_graph))

        self.graphs_callback(graph_1, graph_2)


    def graph_message_callback(self, graph_msg: FullGraph) -> None:
        """
        Callback function to store messages and trigger processing when two messages are received.
        """
        self._graph_queue.append(graph_msg)
        self.get_logger().warn("Received a graph")

        first_graph: str = "graph_1.pkl"
        second_graph: str = "graph_2.pkl"
        path: str = f"graphs/{self._map_name[:-4]}"
        
        if len(self._graph_queue) == 2:
            self.get_logger().warn("Received two graphs. Starting alignment...")

            graph_1: list[tuple[GraphNodeClass, GraphNodeClass]] = self.load_graph_data(os.path.join(path, first_graph))
            graph_2: list[tuple[GraphNodeClass, GraphNodeClass]] = self.load_graph_data(os.path.join(path, second_graph))

            self.graphs_callback(graph_1, graph_2)


    def graphs_callback(self, graph_list_1: list[tuple[GraphNodeClass, GraphNodeClass]],
                        graph_list_2: list[tuple[GraphNodeClass, GraphNodeClass]]) -> None:
        """
        Processes and aligns two received graphs.
        """
        self.get_logger().warn("Processing graphs")

        graph_1: Graph = self._process_graph(graph_list_1)
        graph_2: Graph = self._process_graph(graph_list_2)

        self.get_logger().warn("Aligning graphs")
        self._map_aligner.align_graphs(graph_1, graph_2)
        self.get_logger().warn("Generating maps")
        self._map_aligner.generate_map()
        self._save_graph_data(self._map_aligner.updated_graph)
        self.get_logger().warn("Map generated and saved.")

        message: CommandMessage = CommandMessage()
        message.confirmation = 1

        self._graph_publisher.publish(message)

        self.get_logger().warn("Message sent. Shutting down node.")
        sys.exit(0)


    def _process_graph(self, graph_list: list[tuple[GraphNodeClass, GraphNodeClass]]) -> Graph:
        """
        

        Args:
            graph_list (list[tuple[GraphNodeClass, GraphNodeClass]]): _description_

        Returns:
            Graph: _description_
        """
        edges: list[int] = []

        new_graph: Graph = Graph()

        for node, adjacent in graph_list:
            if np.isnan(node.image).any() or np.isinf(node.image).any():
                self.get_logger().error(f"Node {node.id} has invalid image data!")

            if np.isnan(node.visual_features).any() or np.isinf(node.visual_features).any():
                self.get_logger().error(f"Node {node.id} has invalid features!")

            if np.isnan(node.pose).any() or np.isinf(node.pose).any():
                self.get_logger().error(f"Node {node.id} has invalid pose!")

            edges.append(node.id)
            edges.append(adjacent.id)

            id: int = node.id
            if id not in new_graph.nodes:
                image: np.ndarray = np.array(node.image.flatten().tolist()).reshape(list(node.image.shape)).astype(np.uint8)
                pose: tuple[float, float] = tuple(node.pose)
                features: np.ndarray = np.array(node.visual_features.tolist())
                semantics: np.ndarray = np.array(node.semantics.tolist())
                new_node: GraphNodeClass = GraphNodeClass(id=id, pose=pose, visual_features=features, image=image, semantics=semantics)
                new_graph.nodes[id] = new_node

            id = adjacent.id
            if id not in new_graph.nodes:
                image: np.ndarray = np.array(adjacent.image.flatten().tolist()).reshape(list(adjacent.image.shape)).astype(np.uint8)
                pose: tuple[float, float] = tuple(adjacent.pose)
                features: np.ndarray = np.array(adjacent.visual_features.tolist())
                semantics: np.ndarray = np.array(adjacent.semantics.tolist())
                new_node: GraphNodeClass = GraphNodeClass(id=id, pose=pose, visual_features=features, image=image, semantics=semantics)
                new_graph.nodes[id] = new_node

        for i in range(0, len(edges), 2):
            new_graph.edges.append((edges[i], edges[i + 1]))
        final_edge: tuple[int, int] = (edges[-2], edges[-1])

        if final_edge not in new_graph.edges:
            new_graph.edges.append(final_edge)
        # self.get_logger().warn(f"{new_graph.edges}")
        return new_graph


    def load_graph_data(self, filename: str) -> list[tuple[GraphNodeClass, GraphNodeClass]]:
        """
        Loads the graph and edges data from a pickle file.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Graph data file '{filename}' not found.")

        with open(filename, "rb") as f:
            graph = pickle.load(f)

        return graph


    def _save_graph_data(self, graph: Graph):
        """
        Saves the graph and edges data to a file using pickle (binary format).
        """
        filename: str = "final_graph.pkl"
        filename = os.path.join(f"graphs/{self._map_name[:-4]}", filename)

        with open(filename, "wb") as f:
            pickle.dump(graph, f)

        self.get_logger().warn("Graph saving done with pickle")


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
