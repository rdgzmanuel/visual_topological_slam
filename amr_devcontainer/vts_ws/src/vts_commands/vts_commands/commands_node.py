"""ROS2 node for manual or voice-controlled navigation commands."""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING

import cv2
import rclpy
import speech_recognition as sr
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from vts_commands.commands import Commander

if TYPE_CHECKING:
    from vts_graph_building.node import GraphNodeClass


class CommandNode(Node):
    """ROS2 node that handles manual or voice-controlled navigation commands."""

    _VOICE_TRIGGER: str = "oye silla"
    _VOICE_TIMEOUT: float = 5.0
    _MANUAL_SHUTDOWN_DELAY: float = 3.0

    def __init__(self) -> None:
        """Initialize the CommandNode with parameters and commander."""
        super().__init__("commands")

        self._declare_parameters()
        self._setup_publisher()
        self._initialize_commander()
        self._start_mode()

    def _declare_parameters(self) -> None:
        """Declare and load ROS2 parameters."""
        self._graph_name: str = "final_graph.pkl"

        self.declare_parameter("map_name", "default_value")
        self._map_name: str = (
            self.get_parameter("map_name").get_parameter_value().string_value
        )

        self.declare_parameter("mode", "manual")
        self._mode: str = self.get_parameter("mode").get_parameter_value().string_value

        self.declare_parameter("threshold", 0.0)
        self._threshold: float = (
            self.get_parameter("threshold").get_parameter_value().double_value
        )

        self.declare_parameter("query_sentence", "corridor")
        self._query_sentence: str = (
            self.get_parameter("query_sentence").get_parameter_value().string_value
        )

    def _setup_publisher(self) -> None:
        """Set up ROS2 publishers."""
        self._voice_publisher = self.create_publisher(
            Float32MultiArray, "voice_commands", 10
        )

    def _initialize_commander(self) -> None:
        """Initialize the Commander instance."""
        self._commander: Commander = Commander(
            threshold=self._threshold,
            graph_name=self._graph_name,
            map_name=self._map_name,
        )

    def _start_mode(self) -> None:
        """Start the appropriate operating mode."""
        if self._mode == "manual":
            self._run_manual_mode()
        elif self._mode == "voice":
            self._run_voice_mode()
        else:
            self.get_logger().error(f"Invalid mode: {self._mode}")
            raise ValueError(f"Invalid mode: {self._mode}")

    def _run_manual_mode(self) -> None:
        """Execute one-shot query from the static sentence in manual mode."""
        top_nodes: list[tuple[GraphNodeClass, float]] = self._commander.find_top_nodes(
            self._query_sentence, top_k=3
        )

        if top_nodes:
            for rank, (node, similarity) in enumerate(top_nodes, start=1):
                self.get_logger().warn(
                    f"#{rank} Node {node.id} (sim={similarity:.4f}). Pose: {node.pose}"
                )
                self._save_node_image(node, rank, similarity)
        else:
            self.get_logger().warn("No similar places found.")

        time.sleep(self._MANUAL_SHUTDOWN_DELAY)
        self._request_shutdown()

    def _save_node_image(
        self, node: GraphNodeClass, rank: int, similarity: float
    ) -> None:
        """
        Save the image of a graph node to file.

        Args:
            node: Graph node containing the image to save.
            rank: Ranking position (1 = best match).
            similarity: Similarity score for this node.
        """
        output_dir: str = "images/room_commands"
        os.makedirs(output_dir, exist_ok=True)

        filename: str = f"rank{rank}.png"
        output_path: str = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, node.image)
        self.get_logger().info(f"Saved: {output_path}")

    def _run_voice_mode(self) -> None:
        """Run continuous voice command listening loop."""
        recognizer: sr.Recognizer = sr.Recognizer()

        try:
            microphone: sr.Microphone = sr.Microphone()
        except OSError:
            self.get_logger().error(
                "No microphone input device found. Is audio available?"
            )
            raise RuntimeError("Microphone not available")

        self.get_logger().info(
            f"Voice mode activated. Say '{self._VOICE_TRIGGER}' to issue a command."
        )

        while rclpy.ok():
            self._process_voice_input(recognizer, microphone)

    def _process_voice_input(
        self, recognizer: sr.Recognizer, microphone: sr.Microphone
    ) -> None:
        """
        Process a single voice input cycle.

        Args:
            recognizer: Speech recognition engine.
            microphone: Microphone input source.
        """
        try:
            trigger_detected: bool = self._listen_for_trigger(recognizer, microphone)

            if trigger_detected:
                self._handle_voice_command(recognizer, microphone)

        except sr.UnknownValueError:
            self.get_logger().warn("Could not understand audio.")
        except sr.RequestError as e:
            self.get_logger().error(f"Speech recognition error: {e}")
        except sr.WaitTimeoutError:
            self.get_logger().warn("Voice command timeout.")
        except Exception as ex:
            self.get_logger().error(f"Unexpected error: {ex}")

    def _listen_for_trigger(
        self, recognizer: sr.Recognizer, microphone: sr.Microphone
    ) -> bool:
        """
        Listen for the voice trigger phrase.

        Args:
            recognizer: Speech recognition engine.
            microphone: Microphone input source.

        Returns:
            True if trigger phrase was detected.
        """
        self.get_logger().info("Listening for trigger...")

        with microphone as source:
            audio: sr.AudioData = recognizer.listen(source)

        trigger_phrase: str = recognizer.recognize_google(
            audio, language="es-ES"
        ).lower()
        self.get_logger().info(f"Heard: {trigger_phrase}")

        return self._VOICE_TRIGGER in trigger_phrase

    def _handle_voice_command(
        self, recognizer: sr.Recognizer, microphone: sr.Microphone
    ) -> None:
        """
        Handle a voice command after trigger detection.

        Args:
            recognizer: Speech recognition engine.
            microphone: Microphone input source.
        """
        self.get_logger().info("Trigger detected. Listening for command...")

        with microphone as source:
            audio: sr.AudioData = recognizer.listen(source, timeout=self._VOICE_TIMEOUT)

        command_text: str = recognizer.recognize_google(audio, language="es-ES")
        self.get_logger().info(f"Command received: {command_text}")

        node: GraphNodeClass | None = self._commander.find_closest_node(command_text)

        if node is not None:
            self._publish_navigation_target(node)
        else:
            self.get_logger().warn("No similar place found.")

    def _publish_navigation_target(self, node: GraphNodeClass) -> None:
        """
        Publish navigation target coordinates.

        Args:
            node: Target graph node.
        """
        x: float = float(node.pose[0])
        y: float = float(node.pose[1])
        self.get_logger().info(f"Closest node: {node.id} at ({x}, {y})")

        msg: Float32MultiArray = Float32MultiArray()
        msg.data = [x, y]
        self._voice_publisher.publish(msg)

    def _request_shutdown(self) -> None:
        """Request graceful node shutdown."""
        raise SystemExit(0)


def main(args: list[str] | None = None) -> None:
    """
    Main entry point for the command node.

    Args:
        args: Command line arguments.
    """
    rclpy.init(args=args)

    try:
        command_node: CommandNode = CommandNode()
        rclpy.spin(command_node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        sys.exit(1)
    finally:
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
