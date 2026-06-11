"""Order-invariant fusion of N topological maps (N <= ~5 by design).

The previous implementation integrated map 2 into map 1 sequentially with
fixed distance/similarity thresholds, so the output depended on the order of
the inputs. Here fusion is formulated jointly:

1. Maps are deterministically sorted by their ``frame_id`` so any input
   ordering yields the same internal ordering.
2. The reference frame is the map whose ``frame_id`` sorts first; every other
   map is registered to it with an SE(2) RANSAC over mutual-nearest-neighbor
   visual correspondences (tolerance derived from median node spacing, not a
   constant). If registration fails (too few matches) the identity is used —
   correct for COLD, where all runs share one building frame.
3. All cross-map mutual-NN matches that survive the geometric residual check
   form a correspondence graph over (map, node) pairs; its connected
   components — constrained to at most one node per map, resolved by keeping
   highest-similarity links first — define the fused nodes.
4. Each component is fused with symmetric, order-free operations: descriptor
   = normalized mean, pose = covariance-weighted mean in the reference
   frame, views = union truncated deterministically.

Because every step operates on *sets* with deterministic tie-breaking, the
result is invariant to the order in which maps are provided.
"""

from __future__ import annotations

import numpy as np

from vts_core.matching import (
    estimate_se2_ransac,
    median_spacing,
    mutual_nearest_neighbors,
)
from vts_core.motion import normalize_angle
from vts_core.topo_graph import TopoGraph, TopoNode

NodeRef = tuple[int, int]  # (map index after sorting, node id)


class MultiMapAligner:
    """Joint, order-invariant alignment of several topological maps."""

    def align(self, graphs: list[TopoGraph]) -> TopoGraph:
        """Fuse the given maps into one.

        Args:
            graphs: 2 to ~5 topological maps. Order does not matter.

        Returns:
            The fused TopoGraph, expressed in the reference map's frame.
        """
        if not graphs:
            raise ValueError("No graphs to align")
        if any(not g.nodes for g in graphs):
            raise ValueError("Cannot align empty graphs")
        if len(graphs) == 1:
            return graphs[0]

        ordered: list[TopoGraph] = sorted(graphs, key=lambda g: g.frame_id)
        transforms: list[tuple[np.ndarray, np.ndarray]] = (
            self._register_to_reference(ordered)
        )

        matches: list[tuple[NodeRef, NodeRef, float]] = self._cross_map_matches(
            ordered, transforms
        )
        components: list[set[NodeRef]] = self._consistent_components(
            matches, len(ordered)
        )
        return self._fuse(ordered, transforms, components)

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def _register_to_reference(
        self, ordered: list[TopoGraph]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """SE(2) transform from each map's frame into the reference frame."""
        identity: tuple[np.ndarray, np.ndarray] = (
            np.eye(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        )
        transforms: list[tuple[np.ndarray, np.ndarray]] = [identity]

        ref_ids, ref_features = ordered[0].feature_matrix()
        ref_positions: np.ndarray = ordered[0].positions()
        tolerance: float = median_spacing(ref_positions)

        for graph in ordered[1:]:
            ids, features = graph.feature_matrix()
            positions: np.ndarray = graph.positions()
            pairs: list[tuple[int, int, float]] = mutual_nearest_neighbors(
                features, ref_features
            )
            if len(pairs) < 3:
                transforms.append(identity)
                continue
            source: np.ndarray = np.array(
                [positions[i] for i, _, _ in pairs], dtype=np.float64
            )
            target: np.ndarray = np.array(
                [ref_positions[j] for _, j, _ in pairs], dtype=np.float64
            )
            model: tuple[np.ndarray, np.ndarray] | None = estimate_se2_ransac(
                source, target, inlier_tolerance=tolerance
            )
            transforms.append(model if model is not None else identity)
        return transforms

    @staticmethod
    def _apply(
        transform: tuple[np.ndarray, np.ndarray], pose: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        rotation, translation = transform
        xy: np.ndarray = rotation @ np.array(pose[:2]) + translation
        d_theta: float = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        return (float(xy[0]), float(xy[1]), normalize_angle(pose[2] + d_theta))

    # ------------------------------------------------------------------ #
    # Matching & clustering
    # ------------------------------------------------------------------ #
    def _cross_map_matches(
        self,
        ordered: list[TopoGraph],
        transforms: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[tuple[NodeRef, NodeRef, float]]:
        """Geometrically verified mutual-NN matches for every map pair."""
        all_positions: np.ndarray = np.vstack(
            [
                np.array(
                    [self._apply(transforms[m], node.pose)[:2] for node in g.nodes.values()]
                ).reshape(-1, 2)
                for m, g in enumerate(ordered)
            ]
        )
        tolerance: float = median_spacing(all_positions)

        matches: list[tuple[NodeRef, NodeRef, float]] = []
        for map_a in range(len(ordered)):
            ids_a, feats_a = ordered[map_a].feature_matrix()
            for map_b in range(map_a + 1, len(ordered)):
                ids_b, feats_b = ordered[map_b].feature_matrix()
                for i, j, similarity in mutual_nearest_neighbors(feats_a, feats_b):
                    node_a: TopoNode = ordered[map_a].nodes[ids_a[i]]
                    node_b: TopoNode = ordered[map_b].nodes[ids_b[j]]
                    pos_a: np.ndarray = np.array(
                        self._apply(transforms[map_a], node_a.pose)[:2]
                    )
                    pos_b: np.ndarray = np.array(
                        self._apply(transforms[map_b], node_b.pose)[:2]
                    )
                    if float(np.linalg.norm(pos_a - pos_b)) > tolerance:
                        continue
                    matches.append(
                        ((map_a, ids_a[i]), (map_b, ids_b[j]), similarity)
                    )
        return matches

    def _consistent_components(
        self,
        matches: list[tuple[NodeRef, NodeRef, float]],
        n_maps: int,
    ) -> list[set[NodeRef]]:
        """Greedy union-find honoring 'one node per map per cluster'.

        Matches are processed in deterministic order (similarity descending,
        ties broken by node references), so the clustering is reproducible
        and order-invariant.
        """
        parent: dict[NodeRef, NodeRef] = {}
        members: dict[NodeRef, set[NodeRef]] = {}

        def find(ref: NodeRef) -> NodeRef:
            parent.setdefault(ref, ref)
            members.setdefault(ref, {ref})
            while parent[ref] != ref:
                parent[ref] = parent[parent[ref]]
                ref = parent[ref]
            return ref

        ordered_matches: list[tuple[NodeRef, NodeRef, float]] = sorted(
            matches, key=lambda m: (-m[2], m[0], m[1])
        )
        for ref_a, ref_b, _ in ordered_matches:
            root_a: NodeRef = find(ref_a)
            root_b: NodeRef = find(ref_b)
            if root_a == root_b:
                continue
            maps_a: set[int] = {m for m, _ in members[root_a]}
            maps_b: set[int] = {m for m, _ in members[root_b]}
            if maps_a & maps_b:
                continue  # would put two nodes of one map in a cluster
            parent[root_b] = root_a
            members[root_a] |= members[root_b]

        roots: set[NodeRef] = {find(ref) for ref in parent}
        return [members[root] for root in sorted(roots)]

    # ------------------------------------------------------------------ #
    # Fusion
    # ------------------------------------------------------------------ #
    def _fuse(
        self,
        ordered: list[TopoGraph],
        transforms: list[tuple[np.ndarray, np.ndarray]],
        components: list[set[NodeRef]],
    ) -> TopoGraph:
        fused: TopoGraph = TopoGraph(frame_id=ordered[0].frame_id)

        cluster_of: dict[NodeRef, int] = {}
        clusters: list[set[NodeRef]] = []
        for component in components:
            index: int = len(clusters)
            clusters.append(component)
            for ref in component:
                cluster_of[ref] = index
        # Singletons: every unmatched node is its own cluster.
        for map_index, graph in enumerate(ordered):
            for node_id in sorted(graph.nodes):
                ref: NodeRef = (map_index, node_id)
                if ref not in cluster_of:
                    cluster_of[ref] = len(clusters)
                    clusters.append({ref})

        new_id_of: dict[int, int] = {}
        for cluster_index, cluster in enumerate(clusters):
            refs: list[NodeRef] = sorted(cluster)
            nodes: list[TopoNode] = [ordered[m].nodes[i] for m, i in refs]
            poses: list[tuple[float, float, float]] = [
                self._apply(transforms[m], node.pose)
                for (m, _), node in zip(refs, nodes)
            ]

            weights: np.ndarray = np.array(
                [
                    1.0 / max(float(np.trace(node.pose_covariance)), 1e-6)
                    for node in nodes
                ],
                dtype=np.float64,
            )
            weights = weights / weights.sum()
            x: float = float(sum(w * p[0] for w, p in zip(weights, poses)))
            y: float = float(sum(w * p[1] for w, p in zip(weights, poses)))
            theta: float = float(
                np.arctan2(
                    sum(w * np.sin(p[2]) for w, p in zip(weights, poses)),
                    sum(w * np.cos(p[2]) for w, p in zip(weights, poses)),
                )
            )

            descriptor: np.ndarray = np.stack(
                [node.visual_features for node in nodes]
            ).mean(axis=0)
            descriptor = descriptor / max(float(np.linalg.norm(descriptor)), 1e-12)

            fused_node: TopoNode = TopoNode(
                node_id=cluster_index,
                pose=(x, y, theta),
                visual_features=descriptor.astype(np.float32),
                pose_covariance=min(
                    (node.pose_covariance for node in nodes),
                    key=lambda c: float(np.trace(c)),
                ).copy(),
                room_label=next(
                    (node.room_label for node in nodes if node.room_label), None
                ),
            )
            for node in nodes:
                for view in node.views:
                    fused_node.add_view(view)
            fused.add_node(fused_node)
            new_id_of[cluster_index] = cluster_index

        for map_index, graph in enumerate(ordered):
            for id_a, id_b in graph.edges():
                cluster_a: int = cluster_of[(map_index, id_a)]
                cluster_b: int = cluster_of[(map_index, id_b)]
                if cluster_a != cluster_b:
                    fused.add_edge(cluster_a, cluster_b)

        return fused
