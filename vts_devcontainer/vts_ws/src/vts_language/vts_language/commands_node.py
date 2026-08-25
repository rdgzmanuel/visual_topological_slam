"""Language commands ROS2 node (English-only, text-driven).

Maps a natural-language place description (e.g. "take me to the printer
area") to a node of the topological map and publishes its pose as a
navigation target.

Modes:
- ``manual``: answer the ``query_sentence`` parameter once, write a JSON
  report and the top-k node views to ``output_dir``, then exit. The report
  file is the reliable artifact; the ROS messages are also published and the
  node spins briefly so DDS can deliver them.
- ``topic``: stay alive and answer every std_msgs/String published on
  ``/language/command``. Use this for interactive querying.

Outputs per query:
- ``geometry_msgs/Pose2D`` on ``/language/navigation_target`` (only when the
  answer is confident — see retrieval rejection rule),
- ``std_msgs/String`` (JSON report) on ``/language/result``,
- ``<output_dir>/result.json`` and ``<output_dir>/rank*_node*.pdf`` on disk.

When the retriever flags the answer as ambiguous (top-2 posterior margin too
small) no target is published; the top-k candidates are still reported so the
caller can disambiguate rather than committing to a wrong node.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import suppress

import cv2
import rclpy
from geometry_msgs.msg import Pose2D
from PIL import Image
from rclpy.node import Node
from std_msgs.msg import String

from vts_core.retrieval import PlaceRetriever, SemanticEncoder
from vts_core.topo_graph import TopoGraph, TopoNode


class CommandsNode(Node):
    """Maps natural-language commands to topological-map targets."""

    def __init__(self) -> None:
        super().__init__("commands")

        self.declare_parameter("graph_path", "output/graphs/final_graph.pkl")
        self.declare_parameter("semantic_model", "openai/clip-vit-base-patch32")
        self.declare_parameter("mode", "manual")
        self.declare_parameter("query_sentence", "corridor")
        self.declare_parameter("top_k", 3)
        self.declare_parameter("output_dir", "output/commands")

        graph_path: str = (
            self.get_parameter("graph_path").get_parameter_value().string_value
        )
        model_name: str = (
            self.get_parameter("semantic_model").get_parameter_value().string_value
        )
        self.mode: str = (
            self.get_parameter("mode").get_parameter_value().string_value
        )
        self._query_sentence: str = (
            self.get_parameter("query_sentence").get_parameter_value().string_value
        )
        self._top_k: int = (
            self.get_parameter("top_k").get_parameter_value().integer_value
        )
        self._output_dir: str = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )
        os.makedirs(self._output_dir, exist_ok=True)

        self._target_pub = self.create_publisher(
            Pose2D, "/language/navigation_target", 10
        )
        self._result_pub = self.create_publisher(String, "/language/result", 10)

        graph: TopoGraph = TopoGraph.load(graph_path)
        encoder: SemanticEncoder = SemanticEncoder(model_name)
        self._retriever: PlaceRetriever = PlaceRetriever(encoder, graph)
        self.get_logger().info(
            f"Loaded graph '{graph_path}' "
            f"({len(graph.nodes)} nodes) with model '{model_name}'."
        )

        if self.mode == "topic":
            self.create_subscription(
                String, "/language/command", self._on_command, 10
            )
            self.get_logger().info(
                "Topic mode: publish a query on /language/command "
                "(std_msgs/String)."
            )

    # ------------------------------------------------------------------ #
    def _on_command(self, msg: String) -> None:
        query: str = msg.data.strip()
        if query:
            self.get_logger().info(f"Command: {query}")
            self.answer(query)

    def answer(self, sentence: str) -> dict[str, object]:
        """Answer one query: publish, log, and persist the report to disk."""
        ranked, confident = self._retriever.query(sentence, top_k=self._top_k)

        report: dict[str, object] = {
            "query": sentence,
            "confident": confident,
            "candidates": [
                {
                    "node_id": node.node_id,
                    "posterior": round(posterior, 4),
                    "score": round(score, 4),
                    "pose": list(node.pose),
                    "room_label": node.room_label,
                }
                for node, posterior, score in ranked
            ],
        }

        out: String = String()
        out.data = json.dumps(report)
        self._result_pub.publish(out)
        self.get_logger().info(json.dumps(report, indent=2))

        with open(os.path.join(self._output_dir, "result.json"), "w") as f:
            json.dump(report, f, indent=2)
        for rank, (node, _, _) in enumerate(ranked, start=1):
            self._save_views(node, rank)

        if ranked and confident:
            best: TopoNode = ranked[0][0]
            target: Pose2D = Pose2D()
            target.x = float(best.pose[0])
            target.y = float(best.pose[1])
            target.theta = float(best.pose[2])
            self._target_pub.publish(target)
            self.get_logger().info(
                f"Navigation target: node {best.node_id} at {best.pose}."
            )
        elif ranked:
            self.get_logger().warn(
                "Ambiguous query: posterior margin below bound; reporting "
                "top-k without committing to a target."
            )
        else:
            self.get_logger().warn("No nodes with semantic views in the graph.")
        return report

    def _save_views(self, node: TopoNode, rank: int) -> None:
        for view_index, view in enumerate(node.views):
            path: str = os.path.join(
                self._output_dir,
                f"rank{rank}_node{node.node_id}_v{view_index}.pdf",
            )
            rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(path, "PDF", resolution=150.0)


def _flush(node: CommandsNode, seconds: float = 1.0) -> None:
    """Spin briefly so queued publications are delivered before exit."""
    deadline: float = time.time() + seconds
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    with suppress(KeyboardInterrupt):
        node: CommandsNode = CommandsNode()
        if node.mode == "manual":
            node.answer(node._query_sentence)
            _flush(node)
        elif node.mode == "topic":
            rclpy.spin(node)
        else:
            node.get_logger().error(f"Invalid mode: {node.mode}")
        node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
