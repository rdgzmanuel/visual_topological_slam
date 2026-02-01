"""Commander module for semantic navigation command interpretation."""

from __future__ import annotations

import os
import pickle
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import rclpy.logging
import torch
from deep_translator import GoogleTranslator
from transformers import CLIPModel, CLIPProcessor

if TYPE_CHECKING:
    from vts_graph_building.node import GraphNodeClass
    from vts_map_alignment.graph_class import Graph


class Commander:
    """
    Interpret semantic commands and identify corresponding nodes in a semantic map.

    Uses CLIP-based text embeddings and language translation to enable
    multilingual querying of graph nodes. Text embeddings are generated using
    the same CLIP model used for node image embeddings, ensuring compatible
    embedding spaces for similarity comparison.

    Attributes:
        threshold: Similarity threshold for node matching.
        graph_name: Filename of the graph data.
        map_name: Name of the map used to locate graph files.
    """

    _CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
    _SIMILARITY_EPSILON: float = 1e-10

    # Prompt templates for better CLIP alignment (CLIP was trained on captions)
    _PROMPT_TEMPLATES: list[str] = [
        "a photo of {}",
        "a photo of a room with {}",
        "an indoor scene with {}",
        "{}",
    ]

    # Mapping from common objects/requests to scene descriptions
    _SCENE_MAPPINGS: dict[str, list[str]] = {
        "toilet": ["bathroom", "restroom", "toilet room", "WC"],
        "bathroom": ["bathroom", "restroom", "toilet room"],
        "computer": [
            "computer lab",
            "office with computers",
            "workspace with monitors",
        ],
        "printer": ["printing room", "room with printers", "copy room"],
        "lab": ["laboratory", "research lab", "science lab"],
        "office": ["office", "workspace", "desk area"],
        "kitchen": ["kitchen", "break room", "kitchenette"],
        "meeting": ["meeting room", "conference room"],
        "hallway": ["hallway", "corridor", "passage"],
    }

    def __init__(
        self,
        threshold: float,
        graph_name: str,
        map_name: str,
        use_prompt_templates: bool = False,
        use_scene_mappings: bool = True,
    ) -> None:
        """
        Initialize the Commander.

        Args:
            threshold: Similarity threshold for node matching.
            graph_name: Filename of the graph data.
            map_name: Name of the map used to locate graph files.
            use_prompt_templates: Whether to use CLIP prompt templates.
            use_scene_mappings: Whether to expand queries using scene mappings.
        """
        self._threshold: float = threshold
        self._graph_name: str = graph_name
        self._map_name: str = map_name
        self._use_prompt_templates: bool = use_prompt_templates
        self._use_scene_mappings: bool = use_scene_mappings

        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._logger = rclpy.logging.get_logger("Commander")

        # Use same CLIP model as GraphNodeClass for compatible embeddings
        self._clip_model: CLIPModel = CLIPModel.from_pretrained(
            self._CLIP_MODEL_NAME
        ).to(self._device)
        self._clip_processor: CLIPProcessor = CLIPProcessor.from_pretrained(
            self._CLIP_MODEL_NAME
        )

        # Cache for loaded graph
        self._cached_graph: Graph | None = None
        self._cached_graph_path: str | None = None

    def _get_graph_path(self) -> str:
        """
        Get the full path to the graph file.

        Returns:
            Full path to the graph pickle file.
        """
        return os.path.join(f"graphs/{self._map_name[:-4]}", self._graph_name)

    def _load_graph_data(self) -> Graph:
        """
        Load the graph from pickle file with caching.

        Returns:
            Loaded graph object.

        Raises:
            FileNotFoundError: If graph file does not exist.
        """
        graph_path: str = self._get_graph_path()

        if self._cached_graph is not None and self._cached_graph_path == graph_path:
            return self._cached_graph

        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Graph data file '{graph_path}' not found.")

        with open(graph_path, "rb") as f:
            graph: Graph = pickle.load(f)

        self._cached_graph = graph
        self._cached_graph_path = graph_path

        return graph

    def _translate_to_english(self, text: str) -> str | None:
        """
        Translate text to English using Google Translator.

        Args:
            text: Input text in any language.

        Returns:
            Translated English text, or None if translation fails.
        """
        try:
            return GoogleTranslator(source="auto", target="en").translate(text)
        except Exception as e:
            self._logger.error(f"Translation failed: {e}")
            return None

    def _extract_graph_embeddings(
        self, graph: Graph
    ) -> tuple[list[int], np.ndarray] | None:
        """
        Extract valid node IDs and embeddings from graph.

        Args:
            graph: Graph containing nodes with semantic embeddings.

        Returns:
            Tuple of (node_ids, embedding_matrix) or None if no valid embeddings.
        """
        node_ids: list[int] = []
        embeddings: list[np.ndarray] = []

        for node_id, node in graph.nodes.items():
            if node.semantics is None:
                continue

            node_embedding: np.ndarray = np.asarray(node.semantics)
            if np.linalg.norm(node_embedding) > 0:
                node_ids.append(node_id)
                embeddings.append(node_embedding)

        if not embeddings:
            return None

        embedding_matrix: np.ndarray = np.stack(embeddings)
        return node_ids, embedding_matrix

    def _compute_similarities(
        self,
        query_embedding: np.ndarray,
        embedding_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarities between query and all embeddings.

        Args:
            query_embedding: Query embedding vector.
            embedding_matrix: Matrix of node embeddings (N x D).

        Returns:
            Array of similarity scores.
        """
        query_norm: float = float(np.linalg.norm(query_embedding))
        matrix_norms: np.ndarray = np.linalg.norm(embedding_matrix, axis=1)

        dot_products: np.ndarray = embedding_matrix @ query_embedding
        similarities: np.ndarray = dot_products / (
            matrix_norms * query_norm + self._SIMILARITY_EPSILON
        )

        return similarities

    def find_closest_node(self, query_sentence: str) -> GraphNodeClass | None:
        """
        Find the graph node with highest semantic similarity to query.

        Args:
            query_sentence: Input command or phrase in any language.

        Returns:
            Most similar graph node, or None if no match exceeds threshold.
        """
        top_nodes: list[tuple[GraphNodeClass, float]] = self.find_top_nodes(
            query_sentence, top_k=1
        )
        if top_nodes:
            return top_nodes[0][0]
        return None

    def find_top_nodes(
        self, query_sentence: str, top_k: int = 3
    ) -> list[tuple[GraphNodeClass, float]]:
        """
        Find the top-k graph nodes with highest semantic similarity to query.

        Args:
            query_sentence: Input command or phrase in any language.
            top_k: Number of top matches to return.

        Returns:
            List of (node, similarity) tuples, sorted by descending similarity.
        """
        english_query: str | None = self._translate_to_english(query_sentence)
        if english_query is None:
            return []

        # Generate expanded queries for better matching
        expanded_queries: list[str] = self._expand_query(english_query)
        self._logger.info(f"Expanded queries: {expanded_queries}")

        graph: Graph = self._load_graph_data()

        extraction_result = self._extract_graph_embeddings(graph)
        if extraction_result is None:
            self._logger.error("No valid node embeddings available in the graph.")
            return []

        node_ids, embedding_matrix = extraction_result

        # Track best similarity for each node across all query variations
        node_best_similarities: dict[int, float] = {nid: -1.0 for nid in node_ids}

        for query in expanded_queries:
            query_embedding: np.ndarray = self._obtain_text_embedding(query)
            similarities: np.ndarray = self._compute_similarities(
                query_embedding, embedding_matrix
            )

            for idx, node_id in enumerate(node_ids):
                if similarities[idx] > node_best_similarities[node_id]:
                    node_best_similarities[node_id] = float(similarities[idx])

        # Sort by similarity and get top-k
        sorted_nodes: list[tuple[int, float]] = sorted(
            node_best_similarities.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        # Filter by threshold and build result
        results: list[tuple[GraphNodeClass, float]] = []
        for node_id, similarity in sorted_nodes:
            if similarity > self._threshold:
                results.append((graph.nodes[node_id], similarity))

        # Log results
        self._logger.warn(f"Top {top_k} matches for '{english_query}':")
        for node, sim in results:
            self._logger.warn(f"  Node {node.id}: {sim:.4f}")

        return results

    def _expand_query(self, query: str) -> list[str]:
        """
        Expand a query into multiple variations for better matching.

        Args:
            query: Original query text.

        Returns:
            List of query variations.
        """
        query_lower: str = query.lower().strip()
        expanded: list[str] = []

        # Add scene mappings if enabled
        if self._use_scene_mappings:
            for keyword, scenes in self._SCENE_MAPPINGS.items():
                if keyword in query_lower:
                    expanded.extend(scenes)

        # If no mappings found, use original query
        if not expanded:
            expanded.append(query_lower)

        # Apply prompt templates if enabled
        if self._use_prompt_templates:
            templated: list[str] = []
            for q in expanded:
                for template in self._PROMPT_TEMPLATES:
                    templated.append(template.format(q))
            return templated

        return expanded

    @lru_cache(maxsize=128)
    def _obtain_text_embedding_cached(
        self, normalized_phrase: str
    ) -> tuple[float, ...]:
        """
        Cached version of text embedding computation.

        Args:
            normalized_phrase: Normalized input phrase.

        Returns:
            Embedding as tuple (for hashability).
        """
        inputs = self._clip_processor(
            text=[normalized_phrase], return_tensors="pt", padding=True
        ).to(self._device)

        with torch.no_grad():
            text_features = self._clip_model.get_text_features(**inputs)
            text_features = (
                text_features.pooler_output
                if hasattr(text_features, "pooler_output")
                else text_features
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        embedding: np.ndarray = text_features.cpu().numpy().flatten().astype(np.float32)
        return tuple(embedding.tolist())

    def _obtain_text_embedding(self, text_query: str) -> np.ndarray:
        """
        Convert a text query into a normalized CLIP text embedding.

        Uses the same CLIP model as GraphNodeClass to ensure embeddings
        are in the same space as node image embeddings.

        Args:
            text_query: Description or command text.

        Returns:
            Normalized embedding vector (512-dimensional for CLIP ViT-B/32).
        """
        normalized_phrase: str = text_query.strip().lower()
        cached_tuple: tuple[float, ...] = self._obtain_text_embedding_cached(
            normalized_phrase
        )
        return np.array(cached_tuple, dtype=np.float32)

    def clear_cache(self) -> None:
        """Clear all cached data including graph and embeddings."""
        self._cached_graph = None
        self._cached_graph_path = None
        self._obtain_text_embedding_cached.cache_clear()

    def reload_graph(self) -> None:
        """Force reload of the graph from disk."""
        self._cached_graph = None
        self._cached_graph_path = None
        self._load_graph_data()
