import importlib.util
import unittest

import numpy as np

from vts_core.pose_graph import optimize_se2
from vts_core.topo_graph import TopoGraph, TopoNode


def _node(node_id: int, x: float) -> TopoNode:
    return TopoNode(
        node_id=node_id,
        pose=(x, 0.0, 0.0),
        visual_features=np.array([1.0, 0.0], dtype=np.float32),
    )


class PoseGraphTest(unittest.TestCase):
    def test_optimization_is_monotonic_and_gauge_is_fixed(self) -> None:
        graph = TopoGraph()
        for node_id, x in enumerate((0.0, 1.3, 0.3)):
            graph.add_node(_node(node_id, x))

        odom_info = np.diag([25.0, 25.0, 100.0])
        loop_info = np.diag([4.0, 4.0, 0.0])
        for a, b in ((0, 1), (1, 2)):
            graph.add_edge(a, b)
            graph.set_edge_constraint(
                a, b, (1.0, 0.0, 0.0), odom_info, "odometry"
            )
        graph.add_edge(0, 2)
        graph.set_edge_constraint(
            0, 2, (0.0, 0.0, 0.0), loop_info, "loop"
        )

        fixed_pose = graph.nodes[0].pose
        result = optimize_se2(graph)

        self.assertLessEqual(result.final_error, result.initial_error + 1e-10)
        self.assertEqual(graph.nodes[0].pose, fixed_pose)
        self.assertGreater(result.iterations, 0)

    @unittest.skipUnless(importlib.util.find_spec("gtsam"), "GTSAM not installed")
    def test_gtsam_backend_smoke(self) -> None:
        graph = TopoGraph()
        graph.add_node(_node(0, 0.0))
        graph.add_node(_node(1, 1.2))
        graph.add_edge(0, 1)
        graph.set_edge_constraint(
            0,
            1,
            (1.0, 0.0, 0.0),
            np.diag([25.0, 25.0, 100.0]),
            "odometry",
        )

        result = optimize_se2(graph, backend="gtsam")

        self.assertEqual(result.backend, "gtsam")
        self.assertLessEqual(result.final_error, result.initial_error + 1e-10)
        self.assertEqual(graph.nodes[0].pose, (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
