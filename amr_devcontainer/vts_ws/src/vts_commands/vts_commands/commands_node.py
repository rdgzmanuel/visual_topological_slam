import rclpy
import torch
import pickle
import os
import cv2
import time
import sys
import numpy as np
from deep_translator import GoogleTranslator
from vts_map_alignment.graph_class import Graph
from vts_graph_building.node import GraphNodeClass
from vts_msgs.msg import CommandMessage
from typing import Optional

from rclpy.node import Node
from transformers import CLIPTokenizer, CLIPTextModel


class CommandNode(Node):
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"
    _clip_tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    _clip_text_model: CLIPTextModel = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)

    def __init__(self) -> None:
        super().__init__("commands")

        self._graph_name: str = "final_graph.pkl"

        self.declare_parameter("map_name", "default_value")
        self._map_name: str = self.get_parameter("map_name").get_parameter_value().string_value

        self._graph_subscriber = self.create_subscription(
            CommandMessage, "/commands", self.message_callback, 10
        )

        self._threshold: float = 0.46

        self._query_sentence: str = "Go to the restrooms"

        # self._start_directly()
    

    def _start_directly(self) -> None:
        closest_node: Graph = self._find_closest_node(self._query_sentence)
        if closest_node is not None:
            self.get_logger().warn(f"Closest id {closest_node.id}. Pose: {closest_node.pose}")
            output_file: str = "images/eigenvalues/room_picture.png"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            cv2.imwrite(output_file, closest_node.image)
        else:
            self.get_logger().warn("No similar places found.")

        time.sleep(3)
        sys.exit(0)
    

    def message_callback(self, message: CommandMessage) -> None:
        """
        Callback function to store messages and trigger processing when two messages are received.
        """
        self.get_logger().warn("Received a message")
        
        closest_node: Optional[Graph] = self._find_closest_node(self._query_sentence)

        if closest_node is not None:
            self.get_logger().warn(f"Closest id {closest_node.id}. Pose: {closest_node.pose}. Image: {closest_node.image}")
            output_file: str = "images/eigenvalues/room_picture.png"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            cv2.imwrite(output_file, closest_node.image)
        else:
            self.get_logger().warn("No similar places found.")
        
        time.sleep(3)
        sys.exit(0)

        return None


    def _obtain_embedding(self, context_phrase: str) -> np.ndarray:
        """
        Encodes a context phrase into an embedding vector using HuggingFace CLIP.

        Args:
            context_phrase (str): The description of the room.

        Returns:
            np.ndarray: Embedding vector of the phrase.
        """
        normalized_phrase: str = context_phrase.strip().lower()
        inputs = self._clip_tokenizer([normalized_phrase], return_tensors="pt", padding=True).to(self._device)
        with torch.no_grad():
            outputs = self._clip_text_model(**inputs)
            
            embeddings: torch.Tensor = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True) # shape 512

        return embeddings.cpu().numpy().squeeze()


    def _load_graph_data(self, filename: str) -> Graph:
        """
        Loads the graph and edges data from a pickle file.
        """
        filename = os.path.join(f"graphs/{self._map_name[:-4]}", filename)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Graph data file '{filename}' not found.")

        with open(filename, "rb") as f:
            graph = pickle.load(f)

        return graph
    

    def _find_closest_node(self, query_sentence: str) -> Optional[GraphNodeClass]:
        """
        Finds the graph node whose semantic embedding is most similar to the embedding
        of a given query sentence. The input can be in any language; it will be translated
        to English before processing.

        Args:
            query_sentence (str): The input command or phrase in any language.

        Returns:
            Optional[GraphNodeClass]: The most similar node, or None if no match is above threshold.
        """
        try:
            # Translate input to English using auto-detected source language
            english_query: str = GoogleTranslator(source='auto', target='en').translate(query_sentence)
        except Exception as e:
            self.get_logger().error(f"Translation failed: {e}")
            return None

        query_embedding: np.ndarray = self._obtain_embedding(english_query)

        graph: Graph = self._load_graph_data(self._graph_name)

        node_ids: list[int] = []
        embeddings: list[np.ndarray] = []

        for node_id, node in graph.nodes.items():
            if node.semantics is not None:
                node_embedding = np.asarray(node.semantics)
                if np.linalg.norm(node_embedding) > 0:
                    node_ids.append(node_id)
                    embeddings.append(node_embedding)

        if not embeddings:
            raise ValueError("No valid node embeddings available in the graph.")

        embedding_matrix: np.ndarray = np.stack(embeddings)  # shape: (N, d)
        matrix_norms: np.ndarray = np.linalg.norm(embedding_matrix, axis=1)

        def compute_similarity(query_embedding: np.ndarray) -> np.ndarray:
            query_norm: float = np.linalg.norm(query_embedding)
            dot_products: np.ndarray = embedding_matrix @ query_embedding
            return dot_products / (matrix_norms * query_norm + 1e-10)

        similarities: np.ndarray = compute_similarity(query_embedding)
        max_sim: float = float(np.max(similarities))

        self.get_logger().warn(f"{max_sim}")
        self.get_logger().warn(f"{similarities}")

        if max_sim > self._threshold:
            best_index: int = int(np.argmax(similarities))
            best_node_id: int = node_ids[best_index]
            return graph.nodes[best_node_id]
        else:
            return None


def main(args: list[str] = None) -> None:
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
