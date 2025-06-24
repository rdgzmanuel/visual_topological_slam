import torch
import rclpy
import pickle
import os
import base64
import json
import cv2
import numpy as np
from deep_translator import GoogleTranslator
from vts_map_alignment.graph_class import Graph
from vts_graph_building.node import GraphNodeClass
from typing import Optional
from transformers import CLIPTokenizer, CLIPTextModel


class Commander:
    """
    Commander class is responsible for interpreting semantic commands and identifying
    corresponding nodes in a prebuilt semantic map graph. It leverages CLIP-based
    text embeddings and language translation to enable multilingual querying.
    
    Attributes:
        _map_name (str): Name of the map used to locate graph files.
        _graph_name (str): Filename of the graph data.
        _clip_tokenizer: Tokenizer compatible with CLIP text model.
        _clip_text_model: CLIP text model for obtaining embeddings.
        _device: Torch device (e.g., "cpu" or "cuda").
        _threshold (float): Similarity threshold for node matching.
        get_logger (Callable): Logging method.
    """

    def __init__(self, threshold: float, query_sentence: str, graph_name: str, map_name: str):
        # Initialize required attributes externally before use
        self._threshold: float = threshold
        self._query_sentence: str = query_sentence
        self._graph_name: str = graph_name
        self._map_name: str = map_name

        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._clip_tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_text_model: CLIPTextModel = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(self._device)

        self._logger = rclpy.logging.get_logger('Commander')


    def _load_graph_data(self, filename: str) -> Graph:
        """
        Loads the graph and its nodes from a pickle file.

        Args:
            filename (str): Name of the pickle file containing the graph.

        Returns:
            Graph: Loaded graph object.
        """
        filename = os.path.join(f"graphs/{self._map_name[:-4]}", filename)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Graph data file '{filename}' not found.")

        with open(filename, "rb") as f:
            graph = pickle.load(f)

        return graph
    

    def load_graph_from_json(filename: str) -> Graph:
        with open(filename, 'r') as f:
            graph_data = json.load(f)

        graph = Graph()
        graph.node_id = graph_data["node_id"]
        graph.edges = [tuple(edge) for edge in graph_data["edges"]]

        id_to_node = {}

        for node_info in graph_data["nodes"]:
            visual = decode_array(node_info["visual_features"]["data"],
                                tuple(node_info["visual_features"]["shape"]),
                                node_info["visual_features"]["dtype"])

            semantics = decode_array(node_info["semantics"]["data"],
                                    tuple(node_info["semantics"]["shape"]),
                                    node_info["semantics"]["dtype"])

            image = decode_image(node_info["image"])

            node = GraphNodeClass(id=node_info["id"],
                                pose=tuple(node_info["pose"]),
                                visual_features=visual,
                                image=image,
                                semantics=semantics)

            graph.nodes[node.id] = node
            id_to_node[node.id] = node

        # Set current node
        current_id = graph_data["current_node_id"]
        graph.current_node = id_to_node.get(current_id, None)

        return graph


    def find_closest_node(self, query_sentence: str) -> Optional[GraphNodeClass]:
        """
        Finds the graph node with the highest semantic similarity to the query sentence.

        Args:
            query_sentence (str): The input command or phrase in any language.

        Returns:
            Optional[GraphNodeClass]: The most similar graph node, or None if no match
            exceeds the similarity threshold.
        """
        try:
            english_query: str = GoogleTranslator(source='auto', target='en').translate(query_sentence)
        except Exception as e:
            self._logger().error(f"Translation failed: {e}")
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

        embedding_matrix: np.ndarray = np.stack(embeddings)
        matrix_norms: np.ndarray = np.linalg.norm(embedding_matrix, axis=1)

        def compute_similarity(query_embedding: np.ndarray) -> np.ndarray:
            query_norm: float = np.linalg.norm(query_embedding)
            dot_products: np.ndarray = embedding_matrix @ query_embedding
            return dot_products / (matrix_norms * query_norm + 1e-10)

        similarities: np.ndarray = compute_similarity(query_embedding)
        max_sim: float = float(np.max(similarities))

        self._logger.warn(f"Max similarity: {max_sim}")
        self._logger.warn(f"All similarities: {similarities}")

        if max_sim > self._threshold:
            best_index: int = int(np.argmax(similarities))
            best_node_id: int = node_ids[best_index]
            return graph.nodes[best_node_id]
        else:
            return None


    def _obtain_embedding(self, context_phrase: str) -> np.ndarray:
        """
        Converts a context phrase into a normalized semantic embedding using CLIP.

        Args:
            context_phrase (str): Description or name of the location.

        Returns:
            np.ndarray: A normalized embedding vector.
        """
        normalized_phrase: str = context_phrase.strip().lower()
        inputs = self._clip_tokenizer([normalized_phrase], return_tensors="pt", padding=True).to(self._device)

        with torch.no_grad():
            outputs = self._clip_text_model(**inputs)
            embeddings: torch.Tensor = outputs.last_hidden_state.mean(dim=1)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        return embeddings.cpu().numpy().squeeze()


def decode_array(data: str, shape: tuple, dtype: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(data), dtype=dtype).reshape(shape)

def decode_image(data: str) -> np.ndarray:
    buffer = base64.b64decode(data)
    return cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)