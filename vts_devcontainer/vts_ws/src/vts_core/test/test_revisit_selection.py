import unittest

import numpy as np

from vts_core.mapper import TopologicalMapper
from vts_core.matching import gaussian_position_nll
from vts_core.topo_graph import TopoNode


def _feature(score: float, dimension: int = 8) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    vector[0] = score
    vector[1] = np.sqrt(max(1.0 - score**2, 0.0))
    return vector


def _node(
    node_id: int,
    score: float,
    pose_x: float,
    covariance_scale: float,
    dimension: int = 8,
) -> TopoNode:
    descriptor = _feature(score, dimension)
    return TopoNode(
        node_id=node_id,
        pose=(pose_x, 0.0, 0.0),
        visual_features=descriptor,
        view_features=descriptor[None, :],
        pose_covariance=np.eye(2) * covariance_scale,
    )


def _basis_node(
    node_id: int,
    basis_index: int,
    pose_x: float,
    covariance_scale: float,
    dimension: int = 16,
) -> TopoNode:
    descriptor = np.zeros(dimension, dtype=np.float32)
    descriptor[basis_index] = 1.0
    return TopoNode(
        node_id=node_id,
        pose=(pose_x, 0.0, 0.0),
        visual_features=descriptor,
        view_features=descriptor[None, :],
        pose_covariance=np.eye(2) * covariance_scale,
    )


def _connect_sequence(mapper: TopologicalMapper) -> None:
    for node_id in sorted(mapper.graph.nodes)[1:]:
        mapper.graph.add_edge(node_id - 1, node_id)
        mapper.graph.edge_types[(node_id - 1, node_id)] = "odometry"


class RevisitSelectionTest(unittest.TestCase):
    def test_multi_view_similarity_rewards_viewpoint_coverage(self) -> None:
        first = _node(0, 1.0, 0.0, 0.0)
        second = _node(1, 1.0, 0.0, 0.0)
        first.view_features = np.array([[1.0, 0.0], [0.0, 1.0]])
        second.view_features = np.array([[1.0, 0.0], [0.0, 1.0]])
        complete = TopologicalMapper._node_similarity(first, second)

        second.view_features = np.array([[1.0, 0.0]])
        partial = TopologicalMapper._node_similarity(first, second)

        self.assertAlmostEqual(complete, 1.0)
        self.assertLess(partial, complete)

    def test_sequence_similarity_supports_reverse_traversal(self) -> None:
        mapper = TopologicalMapper(optimize=False)
        # Historical traversal around node 2: node 2 -> 3 -> 4.
        bases = [5, 4, 1, 2, 3, 3, 2]
        for node_id, basis in enumerate(bases):
            mapper.graph.add_node(
                _basis_node(node_id, basis, float(node_id), 0.05 * (node_id + 1))
            )
        _connect_sequence(mapper)
        mapper.current_node_id = 6

        # The doorway itself looks entirely different (basis 0 vs basis 1),
        # but the two preceding live nodes match historical nodes 3 and 4 in
        # reverse temporal order.
        candidate = _basis_node(7, 0, 2.0, 0.4)
        direct = mapper._node_similarity(candidate, mapper.graph.nodes[2])
        sequence = mapper._sequence_similarity(candidate, 2)

        self.assertAlmostEqual(direct, 0.0)
        self.assertAlmostEqual(sequence, 2.0 / 3.0)

    def test_odometry_first_recovers_match_outside_visual_top_five(self) -> None:
        mapper = TopologicalMapper(optimize=False, visual_outlier_k=1.0)

        # Six appearance aliases outrank the true place by direct similarity.
        for node_id in range(6):
            mapper.graph.add_node(
                _node(
                    node_id, 0.8, float(node_id),
                    0.05 * (node_id + 1), dimension=16,
                )
            )
        # Historical context 6 -> 7 -> target 8.
        mapper.graph.add_node(_basis_node(6, 3, 6.0, 0.35))
        mapper.graph.add_node(_basis_node(7, 2, 7.0, 0.40))
        mapper.graph.add_node(_basis_node(8, 1, 8.0, 0.45))
        # Current traversal approaches the target in the reverse direction.
        mapper.graph.add_node(_basis_node(9, 3, 9.0, 0.50))
        mapper.graph.add_node(_basis_node(10, 2, 10.0, 0.55))
        _connect_sequence(mapper)
        mapper.current_node_id = 10

        candidate = _basis_node(11, 0, 8.0, 0.60)
        eligible = list(range(9))
        direct = [
            mapper._node_similarity(candidate, mapper.graph.nodes[node_id])
            for node_id in eligible
        ]
        direct_rank = list(np.argsort(-np.asarray(direct))).index(8)
        self.assertGreaterEqual(direct_rank, 5)

        accepted, proposed, similarity, reason = mapper._find_revisit(candidate)

        self.assertEqual(proposed, 8)
        self.assertEqual(accepted, 8)
        self.assertAlmostEqual(similarity, 2.0 / 3.0)
        self.assertEqual(reason, "accepted")

    def test_unique_geometric_candidate_does_not_require_visual_outlier(self) -> None:
        mapper = TopologicalMapper(optimize=False, visual_outlier_k=2.0)

        # Every node looks identical, so no candidate can be a visual outlier.
        # Geometry nevertheless identifies node 0 unambiguously.
        for node_id in range(7):
            mapper.graph.add_node(
                _basis_node(node_id, 0, 10.0 * node_id, 0.1)
            )
        _connect_sequence(mapper)
        mapper.current_node_id = 6

        candidate = _basis_node(7, 0, 0.0, 0.1)
        accepted, proposed, similarity, reason = mapper._find_revisit(candidate)

        self.assertEqual(proposed, 0)
        self.assertEqual(accepted, 0)
        self.assertAlmostEqual(similarity, 1.0)
        self.assertEqual(reason, "accepted")

    def test_interval_covariance_uses_only_new_uncertainty(self) -> None:
        earlier = np.diag([1.0, 2.0, 3.0])
        later = np.diag([1.2, 2.5, 3.1])
        interval = TopologicalMapper._interval_covariance(earlier, later)
        np.testing.assert_allclose(interval, np.diag([0.2, 0.5, 0.1]))

    def test_node_extent_is_not_treated_as_localization_uncertainty(self) -> None:
        mapper = TopologicalMapper(optimize=False)
        matched = _node(0, 1.0, 0.0, 0.0)
        candidate = _node(1, 1.0, 3.0, 0.001)
        matched.extent_covariance = np.eye(2) * 100.0
        candidate.extent_covariance = np.eye(2) * 100.0

        covariance = mapper._revisit_covariance(candidate, matched)

        np.testing.assert_allclose(
            covariance, np.eye(2) * 0.251, atol=1e-12
        )
        self.assertFalse(
            mapper._passes_gates(np.array([3.0, 0.0]), covariance)
        )

    def test_gaussian_score_penalizes_uninformative_covariance(self) -> None:
        delta = np.array([0.4, 0.0])
        informative = gaussian_position_nll(delta, np.eye(2) * 0.25)
        uninformative = gaussian_position_nll(delta, np.eye(2) * 25.0)
        self.assertLess(informative, uninformative)


if __name__ == "__main__":
    unittest.main()
