import rclpy
import torch
import os
import cv2
import time
import sys
import numpy as np
import speech_recognition as sr
from deep_translator import GoogleTranslator
from vts_graph_building.node import GraphNodeClass
from vts_commands.commands import Commander
from std_msgs.msg import Float32MultiArray
from typing import Optional
from rclpy.node import Node


class CommandNode(Node):
    """
    ROS2 node that handles manual or voice-controlled navigation commands.
    """


    def __init__(self) -> None:
        super().__init__("commands")

        self._graph_name: str = "final_graph.pkl"

        self.declare_parameter("map_name", "default_value")
        self._map_name: str = self.get_parameter("map_name").get_parameter_value().string_value

        self.declare_parameter("mode", "manual")
        self.mode: str = self.get_parameter("mode").get_parameter_value().string_value

        self._threshold: float = 0.46
        self._query_sentence: str = "Go to the toilet"

        self._voice_publisher = self.create_publisher(Float32MultiArray, "voice_commands", 10)

        self.commander: Commander = Commander(
            threshold=self._threshold,
            query_sentence=self._query_sentence,
            graph_name=self._graph_name,
            map_name=self._map_name
        )

        if self.mode == "manual":
            self._run_manual_mode()
        elif self.mode == "voice":
            self._run_voice_mode()
        else:
            self.get_logger().error(f"Invalid mode: {self.mode}")
            sys.exit(1)


    def _run_manual_mode(self) -> None:
        """
        Executes one-shot query from the static sentence when in manual mode.
        """
        closest_node: Optional[GraphNodeClass] = self.commander.find_closest_node(self._query_sentence)
        if closest_node is not None:
            self.get_logger().warn(f"Closest id {closest_node.id}. Pose: {closest_node.pose}")
            output_file: str = "images/eigenvalues/room_picture.png"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            cv2.imwrite(output_file, closest_node.image)
        else:
            self.get_logger().warn("No similar places found.")

        time.sleep(3)
        sys.exit(0)


    def _run_voice_mode(self) -> None:
        """
        Runs in loop listening for voice commands after the "Oye, silla" trigger.
        """
        try:
            mic: sr.Microphone = sr.Microphone()
        except OSError:
            self.get_logger().error("No microphone input device found. Is audio available?")
            sys.exit(1)
    
        recognizer: sr.Recognizer = sr.Recognizer()
        mic: sr.Microphone = sr.Microphone()

        self.get_logger().info("Voice mode activated. Say 'Oye, silla' to issue a command.")

        while rclpy.ok():
            try:
                with mic as source:
                    self.get_logger().info("Listening for trigger...")
                    audio = recognizer.listen(source)

                trigger_phrase: str = recognizer.recognize_google(audio, language="es-ES").lower()
                self.get_logger().info(f"Heard: {trigger_phrase}")

                if "oye silla" in trigger_phrase:
                    self.get_logger().info("Trigger detected. Listening for command...")

                    with mic as source:
                        audio = recognizer.listen(source, timeout=5)

                    # We could combine different languages. For now it'sonly Spanish.
                    command_text: str = recognizer.recognize_google(audio, language="es-ES")
                    self.get_logger().info(f"Command received: {command_text}")

                    node: Optional[GraphNodeClass] = self.commander.find_closest_node(command_text)
                    if node is not None:
                        x: float = float(node.pose[0])
                        y: float = float(node.pose[1])
                        self.get_logger().info(f"Closest node: {node.id} at ({x}, {y})")

                        msg: Float32MultiArray = Float32MultiArray()
                        msg.data = [x, y]
                        self._voice_publisher.publish(msg)
                    else:
                        self.get_logger().warn("No similar place found.")

            except sr.UnknownValueError:
                self.get_logger().warn("Could not understand audio.")
            except sr.RequestError as e:
                self.get_logger().error(f"Speech recognition error: {e}")
            except Exception as ex:
                self.get_logger().error(f"Unexpected error: {ex}")


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    command_node: CommandNode = CommandNode()

    try:
        rclpy.spin(command_node)
    except KeyboardInterrupt:
        pass

    command_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
