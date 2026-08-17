import unittest

import numpy as np

from vts_core.mapper import FrameRecord, TopologicalMapper


def _frame(x: float, descriptor: np.ndarray) -> FrameRecord:
    return FrameRecord(
        descriptor=descriptor,
        pose=(x, 0.0, 0.0),
        covariance=np.diag([0.2, 0.2, 0.1]),
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        gt_pose=(x, 0.0, 0.0),
        room_label="CR",
    )


class MapperOdometryTest(unittest.TestCase):
    def test_confirmed_valley_splits_at_original_boundary(self) -> None:
        class Monitor:
            eigenvalues = []

            def update(self, descriptor):
                return 0.5

        class Detector:
            warmup = 0
            latencies = []

            def __init__(self):
                self.calls = 0

            def step(self, value):
                self.calls += 1
                return 2 if self.calls == 5 else None

        mapper = TopologicalMapper(optimize=False)
        mapper._monitor = Monitor()
        mapper._detector = Detector()
        descriptor = np.array([1.0, 0.0], dtype=np.float32)
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        for index in range(5):
            mapper.process_frame(
                image,
                descriptor,
                (float(index), 0.0, 0.0),
                np.diag([0.1, 0.1, 0.01]),
            )

        self.assertEqual(len(mapper.graph.nodes), 1)
        self.assertEqual([frame.frame_index for frame in mapper._pending_frames], [4])
        mapper.finalize_nodes()
        self.assertEqual(len(mapper.graph.nodes), 2)

    def test_loop_factor_does_not_snap_pose_or_reset_covariance(self) -> None:
        mapper = TopologicalMapper(optimize=False)
        descriptor = np.array([1.0, 0.0], dtype=np.float32)

        mapper._create_node([_frame(0.0, descriptor)])

        mapper._find_revisit = lambda candidate: (
            None,
            None,
            None,
            "no_distinctive_match",
        )
        mapper._create_node([_frame(5.0, descriptor)])

        mapper._find_revisit = lambda candidate: (0, 0, 0.95, "accepted")
        mapper._create_node([_frame(0.5, descriptor)])

        query = mapper.graph.nodes[2]
        self.assertEqual(query.pose, (0.5, 0.0, 0.0))
        np.testing.assert_allclose(query.pose_covariance, np.diag([0.2, 0.2]))
        self.assertEqual(mapper.graph.edge_types[(1, 2)], "odometry")
        self.assertEqual(mapper.graph.edge_types[(0, 2)], "loop")
        self.assertEqual(mapper.current_node_id, 2)

    def test_final_segment_is_flushed(self) -> None:
        mapper = TopologicalMapper(window_size=3, optimize=False)
        descriptor = np.array([1.0, 0.0], dtype=np.float32)
        image = np.zeros((2, 2, 3), dtype=np.uint8)

        for x in (0.0, 0.1, 0.2):
            mapper.process_frame(
                image,
                descriptor,
                (x, 0.0, 0.0),
                np.diag([0.1, 0.1, 0.01]),
            )

        self.assertEqual(len(mapper.graph.nodes), 0)
        mapper.finalize_nodes()
        self.assertEqual(len(mapper.graph.nodes), 1)
        self.assertEqual(mapper.finalize_nodes(), None)


if __name__ == "__main__":
    unittest.main()
