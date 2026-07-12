"""Unified topological graph data model.

This single representation replaces the previous duality between the
edge-list produced by the graph builder and the ``Graph`` class expected by
the alignment and language modules. It is a plain, picklable structure with
no ROS, torch, or model dependencies, so every package (mapping, alignment,
language, evaluation) can load it safely.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np

Pose2D = tuple[float, float, float]


@dataclass
class TopoNode:
    """A node of the topological map.

    Attributes:
        node_id: Unique identifier within one graph.
        pose: (x, y, theta) in the odometry frame of the run that created it.
        visual_features: L2-normalized place-recognition descriptor.
        views: Up to ``MAX_VIEWS`` representative BGR images of the place.
            Multiple raw views replace the old SIFT-stitched panoramas, which
            frequently degenerated and polluted both the descriptors and the
            CLIP semantics.
        view_embeddings: One semantic (CLIP/SigLIP) embedding per view,
            shape (n_views, d). Filled lazily by the semantic encoder.
        pose_covariance: 2x2 covariance of (x, y) accumulated from odometry
            at creation time. Used for probabilistic revisit gating instead
            of fixed metric thresholds.
        room_label: Optional ground-truth room label (evaluation only).
        gt_pose: Optional ground-truth (x, y, theta) of the *representative*
            frame that fixed this node's pose (the segment medoid), attached
            for evaluation only and NEVER used by the mapping algorithm. It
            must reference the same frame as ``pose`` so that placement error
            measures odometric distortion rather than segmentation choice.
    """

    node_id: int
    pose: Pose2D
    visual_features: np.ndarray
    views: list[np.ndarray] = field(default_factory=list)
    view_embeddings: np.ndarray | None = None
    pose_covariance: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 2), dtype=np.float64)
    )
    room_label: str | None = None
    gt_pose: Pose2D | None = None

    MAX_VIEWS: int = 3

    def add_view(self, image: np.ndarray) -> None:
        """Add a representative view, keeping at most ``MAX_VIEWS`` diverse ones.

        Keeps the first, the most recent, and the middle view, which gives a
        cheap diversity heuristic without storing every frame.

        Args:
            image: BGR image to add.
        """
        if image is None:
            return
        if len(self.views) < self.MAX_VIEWS:
            self.views.append(image)
            return
        # first / middle / last policy: replace the last slot, promote the
        # previous last to the middle slot.
        self.views[1] = self.views[2]
        self.views[2] = image
        # Invalidate cached embeddings: views changed.
        self.view_embeddings = None


class TopoGraph:
    """Undirected topological graph with dict-based adjacency.

    Attributes:
        nodes: Mapping node_id -> TopoNode.
        adjacency: Mapping node_id -> set of neighbor node_ids.
        frame_id: Identifier of the run / coordinate frame this graph lives in.
    """

    def __init__(self, frame_id: str = "map") -> None:
        self.nodes: dict[int, TopoNode] = {}
        self.adjacency: dict[int, set[int]] = {}
        # Relative SE(2) measurement per edge, captured AT EDGE-CREATION TIME,
        # keyed (min_id, max_id) and oriented min -> max. Without stored
        # measurements, pose-graph optimization built from current estimates
        # is identically a no-op (zero residual by construction).
        self.edge_measurements: dict[tuple[int, int], Pose2D] = {}
        # Translational sigma per edge (same key convention), captured at
        # creation; consumed by single-run and multi-map joint optimization.
        self.edge_sigmas: dict[tuple[int, int], float] = {}
        self.frame_id: str = frame_id
        self._next_id: int = 0

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def new_node_id(self) -> int:
        """Return a fresh node id."""
        node_id: int = self._next_id
        self._next_id += 1
        return node_id

    def add_node(self, node: TopoNode) -> None:
        """Insert a node (id must be unique)."""
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate node id {node.node_id}")
        self.nodes[node.node_id] = node
        self.adjacency.setdefault(node.node_id, set())
        self._next_id = max(self._next_id, node.node_id + 1)

    def add_edge(self, id_a: int, id_b: int) -> None:
        """Insert an undirected edge between two existing nodes."""
        if id_a == id_b:
            return
        if id_a not in self.nodes or id_b not in self.nodes:
            raise KeyError(f"Edge ({id_a}, {id_b}) references unknown node")
        self.adjacency[id_a].add(id_b)
        self.adjacency[id_b].add(id_a)

    def remove_edge(self, id_a: int, id_b: int) -> None:
        """Remove an undirected edge if it exists."""
        self.adjacency.get(id_a, set()).discard(id_b)
        self.adjacency.get(id_b, set()).discard(id_a)
        self.edge_measurements.pop((min(id_a, id_b), max(id_a, id_b)), None)
        self.edge_sigmas.pop((min(id_a, id_b), max(id_a, id_b)), None)

    def set_edge_measurement(
        self, id_a: int, id_b: int, pose_a: Pose2D, pose_b: Pose2D
    ) -> None:
        """Store the relative measurement for edge (id_a, id_b).

        Args:
            id_a, id_b: Edge endpoints.
            pose_a: Pose associated with id_a at measurement time.
            pose_b: Pose associated with id_b at measurement time.
        """
        key: tuple[int, int] = (min(id_a, id_b), max(id_a, id_b))
        first, second = (pose_a, pose_b) if id_a <= id_b else (pose_b, pose_a)
        import numpy as _np

        dx: float = second[0] - first[0]
        dy: float = second[1] - first[1]
        cos_t: float = float(_np.cos(first[2]))
        sin_t: float = float(_np.sin(first[2]))
        local_x: float = cos_t * dx + sin_t * dy
        local_y: float = -sin_t * dx + cos_t * dy
        d_theta: float = float(
            (second[2] - first[2] + _np.pi) % (2.0 * _np.pi) - _np.pi
        )
        self.edge_measurements[key] = (local_x, local_y, d_theta)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def edges(self) -> list[tuple[int, int]]:
        """Return each undirected edge exactly once as (min_id, max_id)."""
        seen: set[tuple[int, int]] = set()
        for node_id, neighbors in self.adjacency.items():
            for neighbor_id in neighbors:
                edge: tuple[int, int] = (
                    min(node_id, neighbor_id),
                    max(node_id, neighbor_id),
                )
                seen.add(edge)
        return sorted(seen)

    def positions(self) -> np.ndarray:
        """Return an (N, 2) array of node (x, y) positions, sorted by id."""
        ids: list[int] = sorted(self.nodes)
        return np.array(
            [self.nodes[i].pose[:2] for i in ids], dtype=np.float64
        ).reshape(-1, 2)

    def feature_matrix(self) -> tuple[list[int], np.ndarray]:
        """Return (sorted node ids, stacked L2-normalized descriptors)."""
        ids: list[int] = sorted(self.nodes)
        features: np.ndarray = np.stack(
            [self.nodes[i].visual_features for i in ids]
        ).astype(np.float32)
        norms: np.ndarray = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-12)
        return ids, features

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Pickle the graph to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> "TopoGraph":
        """Load a pickled graph from disk."""
        with open(path, "rb") as f:
            graph: TopoGraph = pickle.load(f)
        if not isinstance(graph, TopoGraph):
            raise TypeError(f"File {path} does not contain a TopoGraph")
        if not hasattr(graph, "edge_measurements"):  # older pickles
            graph.edge_measurements = {}
        if not hasattr(graph, "edge_sigmas"):
            graph.edge_sigmas = {}
        return graph
