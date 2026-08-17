"""Unified topological graph data model.

This single representation replaces the previous duality between the
edge-list produced by the graph builder and the ``Graph`` class expected by
downstream modules. It is a plain, picklable structure with
no ROS, torch, or model dependencies, so mapping, language and evaluation
code can load it safely.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np

Pose2D = tuple[float, float, float]
# Current node and this many immediate predecessors describe the local
# traversal rather than a loop closure. Shared by mapping and evaluation.
LOOP_TEMPORAL_EXCLUSION_NODES: int = 2


@dataclass
class LoopClosureEvent:
    """One loop-closure decision made at a newly emitted node.

    Ground truth is intentionally absent: evaluation joins ``query_node_id``
    and ``matched_node_id`` with node ground truth after mapping has finished.
    """

    query_node_id: int
    matched_node_id: int | None
    accepted: bool
    similarity: float | None
    reason: str


@dataclass
class TopoNode:
    """A node of the topological map.

    Attributes:
        node_id: Unique identifier within one graph.
        pose: (x, y, theta) in the odometry frame of the run that created it.
        visual_features: L2-normalized place-recognition descriptor.
        view_features: Up to ``MAX_VIEWS`` L2-normalized visual descriptors,
            one per representative view. This small matrix supports robust
            multi-view place matching without retaining every frame.
        views: Up to ``MAX_VIEWS`` representative BGR images of the place.
            Multiple raw views replace the old SIFT-stitched panoramas, which
            frequently degenerated and polluted both the descriptors and the
            CLIP semantics.
        view_embeddings: One semantic (CLIP/SigLIP) embedding per view,
            shape (n_views, d). Filled lazily by the semantic encoder.
        pose_covariance: 2x2 covariance of (x, y) accumulated from odometry
            at creation time. Used for probabilistic revisit gating instead
            of fixed metric thresholds.
        extent_covariance: 2x2 spatial covariance of the camera poses assigned
            to this place. It represents how much two valid observations of
            the same topological node may differ in position.
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
    view_features: np.ndarray | None = None
    view_embeddings: np.ndarray | None = None
    pose_covariance: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 2), dtype=np.float64)
    )
    extent_covariance: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 2), dtype=np.float64)
    )
    room_label: str | None = None
    gt_pose: Pose2D | None = None

    MAX_VIEWS: int = 3

    def set_views(
        self,
        images: list[np.ndarray],
        visual_features: np.ndarray | None = None,
    ) -> None:
        """Replace representative views and their optional visual descriptors.

        The visual descriptors are deliberately separate from
        ``view_embeddings``: the former are DINO place features, while the
        latter are semantic embeddings populated lazily by the language
        module.
        """
        kept_images = [image for image in images if image is not None][
            : self.MAX_VIEWS
        ]
        self.views = kept_images
        self.view_embeddings = None
        if visual_features is None:
            self.view_features = None
            return
        features = np.asarray(visual_features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != len(kept_images):
            raise ValueError(
                "visual_features must have shape (number of views, dimension)"
            )
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        self.view_features = features / np.maximum(norms, 1e-12)

    def add_view(self, image: np.ndarray) -> None:
        """Add a representative view, keeping at most ``MAX_VIEWS`` diverse ones.

        Keeps the first, the most recent, and the middle view, which gives a
        cheap diversity heuristic without storing every frame.

        Args:
            image: BGR image to add.
        """
        if image is None:
            return
        # This legacy convenience method cannot maintain descriptor/image
        # alignment. Call set_views() when visual descriptors are available.
        self.view_features = None
        self.view_embeddings = None
        if len(self.views) < self.MAX_VIEWS:
            self.views.append(image)
            return
        # first / middle / last policy: replace the last slot, promote the
        # previous last to the middle slot.
        self.views[1] = self.views[2]
        self.views[2] = image


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
        # Full SE(2) information matrix and factor type per edge. Odometry
        # edges constrain translation and heading; descriptor-only loop
        # closures constrain co-location but carry zero angular information.
        self.edge_information: dict[tuple[int, int], np.ndarray] = {}
        self.edge_types: dict[tuple[int, int], str] = {}
        # Compatibility field for older serialized maps.
        self.edge_sigmas: dict[tuple[int, int], float] = {}
        self.loop_events: list[LoopClosureEvent] = []
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
        self.edge_information.pop((min(id_a, id_b), max(id_a, id_b)), None)
        self.edge_types.pop((min(id_a, id_b), max(id_a, id_b)), None)
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

    def set_edge_constraint(
        self,
        id_a: int,
        id_b: int,
        measurement: Pose2D,
        information: np.ndarray,
        edge_type: str,
    ) -> None:
        """Store an SE(2) factor oriented from the lower to higher node id."""
        if edge_type not in {"odometry", "loop"}:
            raise ValueError("edge_type must be 'odometry' or 'loop'")
        if id_a >= id_b:
            raise ValueError(
                "edge constraints must use canonical id_a < id_b ordering"
            )
        key = (id_a, id_b)
        info = np.asarray(information, dtype=np.float64)
        if info.shape != (3, 3) or not np.allclose(info, info.T, atol=1e-9):
            raise ValueError("information must be a symmetric 3x3 matrix")
        if float(np.min(np.linalg.eigvalsh(info))) < -1e-9:
            raise ValueError("information must be positive semidefinite")
        self.edge_measurements[key] = tuple(float(v) for v in measurement)
        self.edge_information[key] = info.copy()
        self.edge_types[key] = edge_type

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
        if not hasattr(graph, "edge_information"):
            graph.edge_information = {}
        if not hasattr(graph, "edge_types"):
            graph.edge_types = {}
        if not hasattr(graph, "loop_events"):
            graph.loop_events = []
        for node in graph.nodes.values():
            if not hasattr(node, "extent_covariance"):
                node.extent_covariance = np.zeros((2, 2), dtype=np.float64)
            if not hasattr(node, "view_features"):
                node.view_features = None
        return graph
