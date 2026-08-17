import unittest

import numpy as np

from vts_core.motion import OdometryUncertaintyTracker


class OdometryUncertaintyTest(unittest.TestCase):
    def test_recorded_pose_is_not_modified_and_covariance_grows(self) -> None:
        tracker = OdometryUncertaintyTracker()
        poses = [(2.0, -1.0, 0.2), (3.0, -1.0, 0.2), (4.0, -1.0, 0.2)]
        covariances = [tracker.step(pose) for pose in poses]

        self.assertEqual(poses[-1], (4.0, -1.0, 0.2))
        np.testing.assert_allclose(covariances[0], np.zeros((3, 3)))
        self.assertGreater(float(np.trace(covariances[-1])), 0.0)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariances[-1]) >= -1e-12))
        interval = covariances[-1] - covariances[1]
        self.assertTrue(np.all(np.linalg.eigvalsh(interval) >= -1e-12))
        self.assertLess(float(np.trace(interval)), float(np.trace(covariances[-1])))


if __name__ == "__main__":
    unittest.main()
