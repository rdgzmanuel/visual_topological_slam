import tempfile
import unittest
from pathlib import Path

import numpy as np

from vts_players.cold_odometry import load_cold_odometry


class ColdOdometryTest(unittest.TestCase):
    def test_parse_and_angle_aware_interpolation(self) -> None:
        rows = (
            "0 0 0 10 0 2 0 6 1.0 2.0 0 3.10 0 0\n"
            "0 0 0 11 0 2 0 6 3.0 4.0 0 -3.10 0 0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odom.tdf"
            path.write_text(rows, encoding="utf-8")
            odometry = load_cold_odometry(str(path))

        pose, covariance = odometry.at(10.5)
        np.testing.assert_allclose(pose[:2], [2.0, 3.0])
        self.assertAlmostEqual(abs(pose[2]), np.pi, places=2)
        self.assertEqual(covariance.shape, (3, 3))

    def test_large_interpolation_gap_is_rejected(self) -> None:
        rows = (
            "0 0 0 10 0 2 0 6 0 0 0 0 0 0\n"
            "0 0 0 12 0 2 0 6 1 0 0 0 0 0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "odom.tdf"
            path.write_text(rows, encoding="utf-8")
            odometry = load_cold_odometry(str(path))
        with self.assertRaises(ValueError):
            odometry.at(11.0, max_gap_s=1.0)


if __name__ == "__main__":
    unittest.main()
