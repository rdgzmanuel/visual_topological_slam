import unittest

import numpy as np

from vts_core.node_detection import AdaptiveValleyDetector, ConnectivityMonitor


class NodeDetectionTest(unittest.TestCase):
    def test_affinities_are_nonnegative_and_isolates_are_defined(self) -> None:
        monitor = ConnectivityMonitor(window_size=4)
        self.assertIsNone(monitor.update(np.array([1.0, 0.0])))
        value = monitor.update(np.array([-1.0, 0.0]))

        self.assertIsNotNone(value)
        self.assertGreaterEqual(float(value), 0.0)
        self.assertTrue(np.all(monitor.affinity >= 0.0))
        self.assertTrue(np.isfinite(float(value)))

    def test_warmup_and_zero_mad_have_deterministic_behavior(self) -> None:
        detector = AdaptiveValleyDetector(k=1.0, history=10, warmup=4)
        outputs = [detector.step(v) for v in (1, 1, 1, 1, 0, 0, 1)]

        self.assertTrue(all(v is None for v in outputs[:6]))
        self.assertEqual(outputs[-1], 4)
        self.assertEqual(detector.last_latency_samples, 2)


if __name__ == "__main__":
    unittest.main()
