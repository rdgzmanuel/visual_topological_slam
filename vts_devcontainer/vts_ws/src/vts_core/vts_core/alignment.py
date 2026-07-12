"""Order-invariant fusion of N topological maps (N <= ~5 by design).

Two structural lessons from real COLD data shaped this implementation:

1. **No rigid transform exists between independently drifted maps.** A
   previous version verified cross-map matches by registering maps with a
   global SE(2) RANSAC; on real runs every correct match was rejected (the
   two maps' drift fields differ non-rigidly). Verification is therefore
   *pairwise distance consistency*: a set of matches is accepted if the
   inter-match distances agree between the two maps (rigid only locally,
   tolerant to global deformation).
2. **Fusion is a joint pose-graph optimization.** Each map's edges carry
   at-creation odometric measurements; the verified cross-map matches enter
   as zero-displacement constraints (Huber-robustified). Optimizing the
   joint graph expresses every node in one common frame and lets each map's
   loop structure correct the other's drift — the standard multi-session
   SLAM formulation, replacing the old "register then average" scheme.

Order invariance: maps are deterministically sorted by ``frame_id``; all
match selection uses deterministic tie-breaking; fusion operations are
symmetric set operations.
"""

from __future__ import annotations

import gtsam
import numpy as np
from gtsam import BetweenFactorPose2, Pose2, noiseModel

from vts_core.matching import median_spacing, mutual_nearest_neighbors
from vts_core.motion import normalize_angle
from vts_core.topo_graph import Pose2D, TopoGraph, TopoNode

NodeRef = tuple[int, int]  # (map index after sorting, node id)

_HUBER_K: float = 1.345
_KEY_STRIDE: int = 1_000_000
_MIN_CONSISTENT_MATCHES: int = 2


def _key(map_index: int, node_id: int) -> int:
    return (map_index + 1) * _KEY_STRIDE + node_id


class MultiMapAligner:
    """Joint, order-invariant alignment of several topological maps.

    After :meth:`align`, ``last_stats`` holds diagnostics: per-pair raw
    mutual-NN counts, how many matches survived consistency verification,
    cluster counts, and joint-optimization errors.
    """

    def __init__(self) -> None:
        self.last_stats: dict[str, object] = {}

    def align(self, graphs: list[TopoGraph]) -> TopoGraph:
        """Fuse the given maps into one.

        Args:
            graphs: 2 to ~5 topological maps. Order does not matter.

        Returns:
            The fused TopoGraph, expressed in a common optimized frame
            anchored at the first map (by sorted frame_id).
        """
        if not graphs:
            raise ValueError("No graphs to align")
        if any(not g.nodes for g in graphs):
            raise ValueError("Cannot align empty graphs")
        if len(graphs) == 1:
            return graphs[0]

        self.last_stats = {}
        ordered: list[TopoGraph] = sorted(graphs, key=lambda g: g.frame_id)

        matches: list[tuple[NodeRef, NodeRef, float]] = self._verified_matches(
            ordered
        )
        optimized_poses: dict[NodeRef, Pose2D] = self._joint_optimize(
            ordered, matches
        )
        components: list[set[NodeRef]] = self._consistent_components(
            matches, len(ordered)
        )
        fused: TopoGraph = self._fuse(ordered, optimized_poses, components)

        self.last_stats["fused_clusters"] = len(
            [c for c in components if len(c) > 1]
        )
        self.last_stats["nodes_in"] = sum(len(g.nodes) for g in ordered)
        self.last_stats["nodes_out"] = len(fused.nodes)
        return fused

    # ------------------------------------------------------------------ #
    # Matching with pairwise-consistency verification
    # ------------------------------------------------------------------ #
    def _verified_matches(
        self, ordered: list[TopoGraph]
    ) -> list[tuple[NodeRef, NodeRef, float]]:
        all_positions: np.ndarray = np.vstack(
            [g.positions() for g in ordered]
        )
        base_tolerance: float = median_spacing(all_positions)

        verified: list[tuple[NodeRef, NodeRef, float]] = []
        for map_a in range(len(ordered)):
            ids_a, feats_a = ordered[map_a].feature_matrix()
            pos_a: np.ndarray = ordered[map_a].positions()
            for map_b in range(map_a + 1, len(ordered)):
                ids_b, feats_b = ordered[map_b].feature_matrix()
                pos_b: np.ndarray = ordered[map_b].positions()

                raw: list[tuple[int, int, float]] = mutual_nearest_neighbors(
                    feats_a, feats_b
                )
                self.last_stats.setdefault("cross_map_mutual_nn", {})[
                    f"{map_a}-{map_b}"
                ] = len(raw)
                if len(raw) < _MIN_CONSISTENT_MATCHES:
                    continue

                kept: list[int] = self._max_consistent_subset(
                    raw, pos_a, pos_b, base_tolerance
                )
                self.last_stats.setdefault("consistency_verified", {})[
                    f"{map_a}-{map_b}"
                ] = len(kept)
                for index in kept:
                    i, j, similarity = raw[index]
                    verified.append(
                        ((map_a, ids_a[i]), (map_b, ids_b[j]), similarity)
                    )
        return verified

    @staticmethod
    def _max_consistent_subset(
        raw: list[tuple[int, int, float]],
        pos_a: np.ndarray,
        pos_b: np.ndarray,
        base_tolerance: float,
    ) -> list[int]:
        """Greedy maximum pairwise-distance-consistent subset of matches.

        Two matches (i1,j1), (i2,j2) are consistent when the distance
        between i1,i2 in map A agrees with the distance between j1,j2 in
        map B within ``base_tolerance + 20% of the distance`` — rigid
        locally, tolerant to accumulated deformation at long range.
        Deterministic: seeded from the match with the most consistent
        partners, ties broken by match order.
        """
        n: int = len(raw)
        consistent: np.ndarray = np.zeros((n, n), dtype=bool)
        for x in range(n):
            ix, jx, _ = raw[x]
            for y in range(x + 1, n):
                iy, jy, _ = raw[y]
                d_a: float = float(np.linalg.norm(pos_a[ix] - pos_a[iy]))
                d_b: float = float(np.linalg.norm(pos_b[jx] - pos_b[jy]))
                tolerance: float = base_tolerance + 0.2 * max(d_a, d_b)
                ok: bool = abs(d_a - d_b) <= tolerance
                consistent[x, y] = ok
                consistent[y, x] = ok

        degrees: np.ndarray = consistent.sum(axis=1)
        seed: int = int(np.argmax(degrees))  # ties -> lowest index
        kept: list[int] = [
            x for x in range(n) if x == seed or consistent[seed, x]
        ]
        if len(kept) < _MIN_CONSISTENT_MATCHES:
            return []
        return kept

    # ------------------------------------------------------------------ #
    # Joint pose-graph optimization
    # ------------------------------------------------------------------ #
    def _joint_optimize(
        self,
        ordered: list[TopoGraph],
        matches: list[tuple[NodeRef, NodeRef, float]],
    ) -> dict[NodeRef, Pose2D]:
        all_positions: np.ndarray = np.vstack([g.positions() for g in ordered])
        cross_sigma: float = max(median_spacing(all_positions), 0.25)

        factor_graph: gtsam.NonlinearFactorGraph = gtsam.NonlinearFactorGraph()
        estimates: gtsam.Values = gtsam.Values()

        for map_index, graph in enumerate(ordered):
            # Anchor: strong prior on the first map, weak gauge-fixing prior
            # on the others (keeps unconnected maps well-posed).
            first_id: int = sorted(graph.nodes)[0]
            sigma_prior: float = 1e-6 if map_index == 0 else 25.0
            factor_graph.add(
                gtsam.PriorFactorPose2(
                    _key(map_index, first_id),
                    Pose2(*graph.nodes[first_id].pose),
                    noiseModel.Diagonal.Sigmas(
                        np.array([sigma_prior, sigma_prior, sigma_prior])
                    ),
                )
            )
            for id_a, id_b in graph.edges():
                key: tuple[int, int] = (min(id_a, id_b), max(id_a, id_b))
                measurement: Pose2D | None = graph.edge_measurements.get(key)
                if measurement is None:
                    pose_a: Pose2 = Pose2(*graph.nodes[key[0]].pose)
                    pose_b: Pose2 = Pose2(*graph.nodes[key[1]].pose)
                    relative: Pose2 = pose_a.between(pose_b)
                else:
                    relative = Pose2(*measurement)
                sigma: float = max(graph.edge_sigmas.get(key, 0.5), 0.05)
                base = noiseModel.Diagonal.Sigmas(
                    np.array([sigma, sigma, max(0.05, sigma * 0.5)])
                )
                robust = noiseModel.Robust.Create(
                    noiseModel.mEstimator.Huber.Create(_HUBER_K), base
                )
                factor_graph.add(
                    BetweenFactorPose2(
                        _key(map_index, key[0]), _key(map_index, key[1]),
                        relative, robust,
                    )
                )
            for node_id in sorted(graph.nodes):
                estimates.insert(
                    _key(map_index, node_id), Pose2(*graph.nodes[node_id].pose)
                )

        cross_base = noiseModel.Diagonal.Sigmas(
            np.array([cross_sigma, cross_sigma, np.pi])
        )
        cross_robust = noiseModel.Robust.Create(
            noiseModel.mEstimator.Huber.Create(_HUBER_K), cross_base
        )
        for (map_a, id_a), (map_b, id_b), _ in matches:
            factor_graph.add(
                BetweenFactorPose2(
                    _key(map_a, id_a), _key(map_b, id_b),
                    Pose2(0.0, 0.0, 0.0), cross_robust,
                )
            )

        initial_error: float = float(factor_graph.error(estimates))
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            factor_graph, estimates, gtsam.LevenbergMarquardtParams()
        )
        result: gtsam.Values = optimizer.optimize()
        self.last_stats["joint_optimization_error"] = [
            round(initial_error, 2),
            round(float(factor_graph.error(result)), 2),
        ]

        poses: dict[NodeRef, Pose2D] = {}
        for map_index, graph in enumerate(ordered):
            for node_id in graph.nodes:
                optimized: Pose2 = result.atPose2(_key(map_index, node_id))
                poses[(map_index, node_id)] = (
                    float(optimized.x()),
                    float(optimized.y()),
                    normalize_angle(float(optimized.theta())),
                )
        return poses

    # ------------------------------------------------------------------ #
    # Clustering
    # ------------------------------------------------------------------ #
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
        optimized_poses: dict[NodeRef, Pose2D],
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
        for map_index, graph in enumerate(ordered):
            for node_id in sorted(graph.nodes):
                ref: NodeRef = (map_index, node_id)
                if ref not in cluster_of:
                    cluster_of[ref] = len(clusters)
                    clusters.append({ref})

        for cluster_index, cluster in enumerate(clusters):
            refs: list[NodeRef] = sorted(cluster)
            nodes: list[TopoNode] = [ordered[m].nodes[i] for m, i in refs]
            poses: list[Pose2D] = [optimized_poses[ref] for ref in refs]

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

        for map_index, graph in enumerate(ordered):
            for id_a, id_b in graph.edges():
                cluster_a: int = cluster_of[(map_index, id_a)]
                cluster_b: int = cluster_of[(map_index, id_b)]
                if cluster_a != cluster_b:
                    fused.add_edge(cluster_a, cluster_b)

        return fused
