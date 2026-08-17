import tempfile
import unittest
from pathlib import Path

import numpy as np

from vts_players.cid_sims_data import (
    load_ground_truth,
    load_wheel_odometry,
    quaternion_yaw,
    timestamped_color_images,
)


class CidSimsDataTest(unittest.TestCase):
    def test_quaternion_and_pose_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pose_path = Path(directory) / "pose.txt"
            pose_path.write_text(
                "0.0 0 0 0 0 0 0 1\n"
                "1.0 2 4 0 0 0 0.70710678 0.70710678\n"
            )
            stream = load_ground_truth(str(pose_path))
            pose, covariance = stream.at(0.5, max_gap_s=2.0)
            np.testing.assert_allclose(pose[:2], (1.0, 2.0))
            self.assertAlmostEqual(pose[2], np.pi / 4.0, places=6)
            self.assertIsNone(covariance)
            self.assertAlmostEqual(
                quaternion_yaw(0.0, 0.0, 0.70710678, 0.70710678),
                np.pi / 2.0,
                places=6,
            )

    def test_wheel_uncertainty_is_monotonic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            odom_path = Path(directory) / "odom.txt"
            suffix = " 0 0 0 0 0 0"
            odom_path.write_text(
                "0.0 0 0 0 0 0 0 1" + suffix + "\n"
                "1.0 1 0 0 0 0 0 1" + suffix + "\n"
                "2.0 2 0 0 0 0 0 1" + suffix + "\n"
            )
            stream = load_wheel_odometry(str(odom_path))
            self.assertIsNotNone(stream.covariances)
            traces = [float(np.trace(covariance)) for covariance in stream.covariances]
            self.assertLess(traces[0], traces[1])
            self.assertLess(traces[1], traces[2])

    def test_color_images_are_sorted_numerically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name in ("10.0.png", "2.0.png", "not-a-time.png"):
                (Path(directory) / name).touch()
            samples = timestamped_color_images(directory)
            self.assertEqual([sample[0] for sample in samples], [2.0, 10.0])


if __name__ == "__main__":
    unittest.main()
