"""
Graph alignment ROS2 node for aligning two topological graphs.

This module provides a ROS2 node that receives graph messages, processes them,
and aligns two graphs using the MapAligner class.
"""

from __future__ import annotations

import os
import pickle
import sys
from collections import deque
from contextlib import suppress
from typing import TYPE_CHECKING

import numpy as np
import rclpy
from rclpy.node import Node
from vts_graph_building.node import GraphNodeClass
from vts_msgs.msg import CommandMessage, FullGraph

from vts_map_alignment.graph_class import Graph
from vts_map_alignment.map_alignment import MapAligner

if TYPE_CHECKING:
    from rclpy.publisher import Publisher
    from rclpy.subscription import Subscription


class GraphAlignment(Node):
    """
    ROS2 node for aligning two topological graphs.

    This node subscribes to graph messages, accumulates two graphs,
    and performs alignment using the MapAligner class.

    Attributes:
        _map_name: Name of the map file being processed.
        _graph_subscriber: Subscription to receive graph messages.
        _graph_publisher: Publisher for command messages.
        _graph_queue: Queue storing received graph messages.
        _map_aligner: Instance of MapAligner for graph alignment.
    """

    def __init__(self) -> None:
        """Initialize the GraphAlignment node with parameters and ROS2 interfaces."""
        super().__init__("graph_alignment")

        model_name = self._declare_and_get_string_param("model_name", "default_value")
        trajectory = self._declare_and_get_string_param("trajectory", "default_value")
        self._map_name = self._declare_and_get_string_param("map_name", "default_value")

        origin = self._declare_and_get_int_array_param("origin", (0, 0))
        world_limits = self._declare_and_get_double_array_param(
            "world_limits", (0.0, 0.0, 0.0, 0.0)
        )

        self._graph_subscriber: Subscription = self.create_subscription(
            FullGraph, "/graph_alignment", self.graph_message_callback, 10
        )

        self._graph_publisher: Publisher[CommandMessage] = self.create_publisher(
            CommandMessage, "/commands", 10
        )

        self._graph_queue: deque[FullGraph] = deque(maxlen=2)

        self._map_aligner = MapAligner(
            model_name, trajectory, world_limits, origin, self._map_name
        )

        # Uncomment if two grpahs have already been generated. See graph_builder_node.
        # self._start_directly()

    def _declare_and_get_string_param(self, name: str, default: str) -> str:
        """
        Declare and retrieve a string parameter.

        Args:
            name: Parameter name.
            default: Default value if parameter is not set.

        Returns:
            The parameter value as a string.
        """
        self.declare_parameter(name, default)
        return self.get_parameter(name).get_parameter_value().string_value

    def _declare_and_get_int_array_param(
        self, name: str, default: tuple[int, ...]
    ) -> tuple[int, ...]:
        """
        Declare and retrieve an integer array parameter.

        Args:
            name: Parameter name.
            default: Default value if parameter is not set.

        Returns:
            The parameter value as a tuple of integers.
        """
        self.declare_parameter(name, default)
        return tuple(
            self.get_parameter(name).get_parameter_value().integer_array_value.tolist()
        )

    def _declare_and_get_double_array_param(
        self, name: str, default: tuple[float, ...]
    ) -> tuple[float, ...]:
        """
        Declare and retrieve a double array parameter.

        Args:
            name: Parameter name.
            default: Default value if parameter is not set.

        Returns:
            The parameter value as a tuple of floats.
        """
        self.declare_parameter(name, default)
        return tuple(
            self.get_parameter(name).get_parameter_value().double_array_value.tolist()
        )

    def _get_graph_path(self) -> str:
        """
        Get the path to the graph storage directory.

        Returns:
            Path string to the graphs directory.
        """
        return f"graphs/{self._map_name[:-4]}"

    def _start_directly(self) -> None:
        """
        Start graph alignment directly from saved pickle files.

        This method is useful for debugging when you have already run the graph
        building pipeline and just want to align both graphs.
        """
        first_graph_file = "graph_1.pkl"
        second_graph_file = "graph_2.pkl"
        path = self._get_graph_path()

        graph_1 = self._load_graph_data(os.path.join(path, first_graph_file))
        graph_2 = self._load_graph_data(os.path.join(path, second_graph_file))

        self._process_and_align_graphs(graph_1, graph_2)

    def graph_message_callback(self, graph_msg: FullGraph) -> None:
        """
        Handle incoming graph messages.

        Stores messages in a queue and triggers processing when two messages
        are received.

        Args:
            graph_msg: The received FullGraph message.
        """
        self._graph_queue.append(graph_msg)
        self.get_logger().warn("Received a graph")

        if len(self._graph_queue) == 2:
            self.get_logger().warn("Received two graphs. Starting alignment...")

            path = self._get_graph_path()
            graph_1 = self._load_graph_data(os.path.join(path, "graph_1.pkl"))
            graph_2 = self._load_graph_data(os.path.join(path, "graph_2.pkl"))

            self._process_and_align_graphs(graph_1, graph_2)

    def _process_and_align_graphs(
        self,
        graph_list_1: list[tuple[GraphNodeClass, GraphNodeClass]],
        graph_list_2: list[tuple[GraphNodeClass, GraphNodeClass]],
    ) -> None:
        """
        Process and align two received graphs.

        Args:
            graph_list_1: First graph as a list of node-adjacent pairs.
            graph_list_2: Second graph as a list of node-adjacent pairs.
        """
        self.get_logger().warn("Processing graphs")

        graph_1 = self._build_graph_from_list(graph_list_1)
        graph_2 = self._build_graph_from_list(graph_list_2)

        self.get_logger().warn("Aligning graphs")
        self._map_aligner.align_graphs(graph_1, graph_2)

        self.get_logger().warn("Generating maps")
        self._map_aligner.generate_map()
        self.get_logger().warn("Map generated and saved.")

        message = CommandMessage()
        message.confirmation = 1
        self._graph_publisher.publish(message)

        self.get_logger().warn("Message sent. Shutting down node.")
        sys.exit(0)

    def _build_graph_from_list(
        self, graph_list: list[tuple[GraphNodeClass, GraphNodeClass]]
    ) -> Graph:
        """
        Build a Graph object from a list of node-adjacent pairs.

        Args:
            graph_list: List of tuples where each tuple contains a node and its
                adjacent node.

        Returns:
            A Graph object containing the processed nodes and edges.

        Raises:
            ValueError: If graph_list is empty.
        """
        if not graph_list:
            raise ValueError("Cannot build graph from empty list")

        graph = Graph()
        edges: list[int] = []

        for node, adjacent in graph_list:
            self._validate_node_data(node)

            edges.append(node.id)
            edges.append(adjacent.id)

            self._add_node_to_graph(graph, node)
            self._add_node_to_graph(graph, adjacent)

        self._build_edges_from_list(graph, edges)

        return graph

    def _validate_node_data(self, node: GraphNodeClass) -> None:
        """
        Validate that node data does not contain NaN or Inf values.

        Args:
            node: The node to validate.
        """
        if np.isnan(node.image).any() or np.isinf(node.image).any():
            self.get_logger().error(f"Node {node.id} has invalid image data!")

        if np.isnan(node.visual_features).any() or np.isinf(node.visual_features).any():
            self.get_logger().error(f"Node {node.id} has invalid features!")

        if np.isnan(node.pose).any() or np.isinf(node.pose).any():
            self.get_logger().error(f"Node {node.id} has invalid pose!")

    def _add_node_to_graph(self, graph: Graph, node: GraphNodeClass) -> None:
        """
        Add a node to the graph if not already present.

        Args:
            graph: The graph to add the node to.
            node: The node to add.
        """
        if node.id in graph.nodes:
            return

        image = (
            np
            .array(node.image.flatten().tolist())
            .reshape(list(node.image.shape))
            .astype(np.uint8)
        )
        pose: tuple[float, float] = tuple(node.pose)
        features = np.array(node.visual_features.tolist())
        semantics = np.array(node.semantics.tolist())

        new_node = GraphNodeClass(
            id=node.id,
            pose=pose,
            visual_features=features,
            image=image,
            semantics=semantics,
        )
        graph.nodes[node.id] = new_node

    def _build_edges_from_list(self, graph: Graph, edges: list[int]) -> None:
        """
        Build edges in the graph from a flat list of node IDs.

        Args:
            graph: The graph to add edges to.
            edges: Flat list of node IDs where consecutive pairs form edges.

        Raises:
            ValueError: If edges list has odd length.
        """
        if len(edges) % 2 != 0:
            raise ValueError("Edges list must have even length")

        for i in range(0, len(edges), 2):
            edge = (edges[i], edges[i + 1])
            if edge not in graph.edges:
                graph.edges.append(edge)

    def _load_graph_data(
        self, filename: str
    ) -> list[tuple[GraphNodeClass, GraphNodeClass]]:
        """
        Load graph data from a pickle file.

        Args:
            filename: Path to the pickle file.

        Returns:
            List of node-adjacent pairs.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Graph data file '{filename}' not found.")

        with open(filename, "rb") as f:
            return pickle.load(f)


def main(args: list[str] | None = None) -> None:
    """
    Entry point for the graph alignment node.

    Args:
        args: Command line arguments to pass to rclpy.init.
    """
    rclpy.init(args=args)
    graph_alignment_node: GraphAlignment = GraphAlignment()

    with suppress(KeyboardInterrupt):
        rclpy.spin(graph_alignment_node)

    graph_alignment_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
