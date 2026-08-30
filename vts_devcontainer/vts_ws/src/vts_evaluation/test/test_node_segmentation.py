import tempfile
import unittest
from pathlib import Path

import numpy as np

from vts_evaluation.node_segmentation import (
    FixedValleyDetector,
    _match_boundaries,
    aggregate_results,
    calibrate_fixed_delta,
    detect_valleys,
    load_cold_frame_labels,
    segmentation_metrics,
    valley_indices_to_splits,
)


class NodeSegmentationTest(unittest.TestCase):
    def test_fixed_detector_returns_local_minimum_index(self) -> None:
        signal = np.asarray([1.0, 1.1, 1.0, 0.6, 0.4, 0.7, 1.0])
        valleys = detect_valleys(signal, FixedValleyDetector(delta=0.2, warmup=2))
        self.assertEqual(valleys, [4])

    def test_lambda_index_is_converted_to_mapper_split_position(self) -> None:
        self.assertEqual(valley_indices_to_splits([4, 8], frame_count=12), [6, 10])

    def test_boundary_matching_is_one_to_one(self) -> None:
        self.assertEqual(_match_boundaries([10], [9, 11], tolerance_frames=2), 1)

    def test_semantic_metrics_reward_aligned_splits(self) -> None:
        signal = np.ones(7)
        labels = ["A"] * 4 + ["B"] * 4
        metrics = segmentation_metrics(
            signal,
            valleys=[2],  # split position 4
            labels=labels,
            tolerance_frames=0,
            frame_rate_hz=2.0,
        )
        self.assertEqual(metrics["nodes"], 2)
        self.assertEqual(metrics["transition_recall"], 1.0)
        self.assertEqual(metrics["frame_weighted_node_purity"], 1.0)
        self.assertEqual(metrics["mean_segments_per_semantic_episode"], 1.0)

    def test_fixed_calibration_matches_target_when_grid_contains_solution(self) -> None:
        signal = np.tile(np.asarray([1.0, 0.2, 1.0]), 8)
        calibration = calibrate_fixed_delta(
            signal, target_boundaries=8, warmup=2, candidate_count=256
        )
        self.assertEqual(calibration.fixed_boundaries, 8)

    def test_cold_labels_follow_sorted_image_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "std_cam"
            images.mkdir()
            first = "t1.000000_x0_y0_a0.jpeg"
            second = "t2.000000_x1_y0_a0.jpeg"
            (images / second).touch()
            (images / first).touch()
            labels = root / "places.lst"
            labels.write_text(f"{second} 2PO1-A\n{first} CR-A\n")

            result = load_cold_frame_labels(str(images), str(labels))

        self.assertEqual(result, ["CR", "2PO"])

    def test_aggregate_selects_only_final_adaptive_parameter(self) -> None:
        base = {
            "environment": "case",
            "nodes_per_1000_frames": 10.0,
            "transition_recall": 0.5,
            "frame_weighted_node_purity": 0.8,
            "mean_segments_per_semantic_episode": 1.5,
        }
        rows = [
            {**base, "method": "adaptive", "parameter_value": 1.0, "nodes": 12},
            {**base, "method": "adaptive", "parameter_value": 2.0, "nodes": 10},
            {**base, "method": "fixed", "parameter_value": 0.1, "nodes": 9},
        ]

        summary = aggregate_results(rows, final_k=2.0)

        comparison = summary["final_adaptive_vs_fixed"]
        self.assertEqual(comparison["adaptive"]["total_nodes"], 10)
        self.assertEqual(comparison["fixed"]["total_nodes"], 9)


if __name__ == "__main__":
    unittest.main()
