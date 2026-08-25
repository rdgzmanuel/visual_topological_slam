import unittest

import numpy as np

from vts_core.directional import (
    fit_von_mises_fisher,
    vmf_log_overlap_ratio,
)


class DirectionalStatisticsTest(unittest.TestCase):
    def test_concentration_tracks_directional_consistency(self) -> None:
        concentrated = np.array(
            [[1.0, 0.0, 0.0], [0.99, 0.1, 0.0], [0.99, -0.1, 0.0]]
        )
        dispersed = np.eye(3)

        concentrated_fit = fit_von_mises_fisher(concentrated)
        dispersed_fit = fit_von_mises_fisher(dispersed)

        self.assertGreater(
            concentrated_fit.mean_resultant_length,
            dispersed_fit.mean_resultant_length,
        )
        self.assertGreater(
            concentrated_fit.concentration, dispersed_fit.concentration
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(concentrated_fit.mean_direction)), 1.0, places=6
        )

    def test_single_observation_has_unknown_concentration(self) -> None:
        fitted = fit_von_mises_fisher(np.array([[1.0, 0.0, 0.0]]))

        self.assertEqual(fitted.sample_count, 1)
        self.assertEqual(fitted.concentration, 0.0)

    def test_overlap_is_symmetric_and_neutral_for_uniform_place(self) -> None:
        first = np.array([1.0, 0.0, 0.0, 0.0])
        second = np.array([0.8, 0.6, 0.0, 0.0])

        forward = vmf_log_overlap_ratio(first, 20.0, second, 10.0)
        reverse = vmf_log_overlap_ratio(second, 10.0, first, 20.0)
        neutral = vmf_log_overlap_ratio(first, 0.0, second, 10.0)

        self.assertAlmostEqual(forward, reverse, places=10)
        self.assertAlmostEqual(neutral, 0.0, places=10)

    def test_concentrated_alignment_has_more_evidence_than_mismatch(self) -> None:
        first = np.array([1.0, 0.0, 0.0, 0.0])
        aligned = np.array([0.98, 0.2, 0.0, 0.0])
        orthogonal = np.array([0.0, 1.0, 0.0, 0.0])

        aligned_score = vmf_log_overlap_ratio(first, 50.0, aligned, 50.0)
        mismatch_score = vmf_log_overlap_ratio(
            first, 50.0, orthogonal, 50.0
        )

        self.assertGreater(aligned_score, mismatch_score)


if __name__ == "__main__":
    unittest.main()
