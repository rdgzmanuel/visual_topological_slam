import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vts_core.topo_graph import LoopClosureEvent, TopoGraph, TopoNode
from vts_evaluation.tolerance_sweep import (
    EvaluationCase,
    evaluate_case,
    validate_tolerances,
)


def _node(node_id: int, x: float) -> TopoNode:
    return TopoNode(
        node_id=node_id,
        pose=(x, 0.0, 0.0),
        visual_features=np.array([1.0, 0.0], dtype=np.float32),
    )


class ToleranceSweepTest(unittest.TestCase):
    def test_tolerances_are_positive_finite_and_unique(self) -> None:
        self.assertEqual(validate_tolerances([0.5, 1.0, 0.5]), [0.5, 1.0])
        for invalid in ([0.0], [-1.0], [float("nan")]):
            with self.assertRaises(ValueError):
                validate_tolerances(invalid)

    def test_same_graph_is_rescored_at_each_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            graph_path = root / "graph.pkl"
            node_gt_path = root / "node_gt.json"
            images_path = root / "images"
            images_path.mkdir()

            positions = (0.0, 1.0, 2.0, 0.1)
            graph = TopoGraph()
            for node_id, x in enumerate(positions):
                graph.add_node(_node(node_id, x))
            information = np.eye(3)
            for first, second in ((0, 1), (1, 2), (2, 3)):
                graph.add_edge(first, second)
                graph.set_edge_constraint(
                    first,
                    second,
                    (1.0, 0.0, 0.0),
                    information,
                    "odometry",
                )
            graph.add_edge(0, 3)
            graph.set_edge_constraint(
                0,
                3,
                (0.0, 0.0, 0.0),
                information,
                "loop",
            )
            graph.loop_events.append(
                LoopClosureEvent(3, 0, True, 0.9, "accepted")
            )
            graph.save(str(graph_path))

            node_gt_path.write_text(
                json.dumps(
                    {
                        str(node_id): [x, 0.0, 0.0]
                        for node_id, x in enumerate(positions)
                    }
                )
            )
            for index, x in enumerate(positions):
                (images_path / f"t{index}_x{x}_y0.0_a0.0.jpeg").touch()

            rows = evaluate_case(
                EvaluationCase(
                    "test",
                    str(graph_path),
                    str(node_gt_path),
                    str(images_path),
                ),
                [0.05, 0.5],
            )

        self.assertEqual(rows[0]["loop_tp"], 0)
        self.assertEqual(rows[0]["loop_fp"], 1)
        self.assertEqual(rows[1]["loop_tp"], 1)
        self.assertEqual(rows[1]["loop_fp"], 0)
        self.assertEqual(rows[0]["n_nodes"], rows[1]["n_nodes"])
        self.assertEqual(rows[0]["n_loop_edges"], rows[1]["n_loop_edges"])


if __name__ == "__main__":
    unittest.main()
