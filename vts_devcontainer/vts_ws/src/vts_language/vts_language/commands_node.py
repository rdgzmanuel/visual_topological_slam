"""Language commands ROS2 node (English-only).

Modes:
- ``manual``: answers the ``query_sentence`` parameter once, saves the
  top-k node views for inspection, and reports calibrated posteriors.
- ``voice``: continuous loop — trigger word, then a command — using Google's
  online recognizer (stays online per your requirement).

The navigation target is published as ``geometry_msgs/Pose2D`` on
``/language/navigation_target`` together with a JSON report on
``/language/result``. When the retriever flags the answer as ambiguous
(top-2 posterior margin too small), no target is published and the top-k
candidates are reported instead — propagating uncertainty rather than
silently committing to a wrong node, which was a major failure mode before.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import suppress

import cv2
import rclpy
from geometry_msgs.msg import Pose2D
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
        self.declare_parameter("voice_trigger", "hey robot")
        self.declare_parameter("top_k", 3)
        self.declare_parameter("output_dir", "output/commands")

        graph_path: str = (
            self.get_parameter("graph_path").get_parameter_value().string_value
        )
        model_name: str = (
            self.get_parameter("semantic_model").get_parameter_value().string_value
        )
        self._mode: str = (
            self.get_parameter("mode").get_parameter_value().string_value
        )
        self._query_sentence: str = (
            self.get_parameter("query_sentence").get_parameter_value().string_value
        )
        self._voice_trigger: str = (
            self.get_parameter("voice_trigger").get_parameter_value().string_value
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

        if self._mode == "manual":
            self._answer(self._query_sentence)
            raise SystemExit(0)
        if self._mode == "voice":
            self._voice_loop()
        else:
            raise ValueError(f"Invalid mode: {self._mode}")

    # ------------------------------------------------------------------ #
    def _answer(self, sentence: str) -> None:
        ranked, confident = self._retriever.query(sentence, top_k=self._top_k)

        report: dict[str, object] = {
            "query": sentence,
            "confident": confident,
            "candidates": [
                {
                    "node_id": node.node_id,
                    "posterior": round(posterior, 4),
                    "pose": list(node.pose),
                    "room_label": node.room_label,
                }
                for node, posterior in ranked
            ],
        }
        out: String = String()
        out.data = json.dumps(report)
        self._result_pub.publish(out)
        self.get_logger().info(json.dumps(report, indent=2))

        for rank, (node, posterior) in enumerate(ranked, start=1):
            self._save_views(node, rank, posterior)

        if ranked and confident:
            best: TopoNode = ranked[0][0]
            target: Pose2D = Pose2D()
            target.x = float(best.pose[0])
            target.y = float(best.pose[1])
            target.theta = float(best.pose[2])
            self._target_pub.publish(target)
        elif ranked:
            self.get_logger().warn(
                "Ambiguous query: posterior margin below bound; "
                "reporting top-k without committing to a target."
            )
        else:
            self.get_logger().warn("No nodes with semantic views in the graph.")

    def _save_views(self, node: TopoNode, rank: int, posterior: float) -> None:
        for view_index, view in enumerate(node.views):
            path: str = os.path.join(
                self._output_dir,
                f"rank{rank}_node{node.node_id}_p{posterior:.3f}_v{view_index}.png",
            )
            cv2.imwrite(path, view)

    # ------------------------------------------------------------------ #
    def _voice_loop(self) -> None:
        import speech_recognition as sr  # local import: optional dependency

        recognizer: sr.Recognizer = sr.Recognizer()
        try:
            microphone: sr.Microphone = sr.Microphone()
        except OSError as error:
            self.get_logger().error(f"No microphone available: {error}")
            raise SystemExit(1)

        self.get_logger().info(
            f"Voice mode. Say '{self._voice_trigger}' followed by a command."
        )
        while rclpy.ok():
            try:
                with microphone as source:
                    audio: sr.AudioData = recognizer.listen(source)
                heard: str = recognizer.recognize_google(
                    audio, language="en-US"
                ).lower()
                if self._voice_trigger not in heard:
                    continue
                with microphone as source:
                    audio = recognizer.listen(source, timeout=5.0)
                command: str = recognizer.recognize_google(audio, language="en-US")
                self.get_logger().info(f"Command: {command}")
                self._answer(command)
            except sr.UnknownValueError:
                self.get_logger().warn("Could not understand audio.")
            except sr.RequestError as error:
                self.get_logger().error(f"Speech recognition error: {error}")
            except sr.WaitTimeoutError:
                self.get_logger().warn("Voice command timeout.")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    with suppress(KeyboardInterrupt, SystemExit):
        node: CommandsNode = CommandsNode()
        rclpy.spin(node)
        node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
