"""Evaluate adaptive topological-node segmentation from saved signals.

The experiment replays valley detection over existing lambda_2 traces, so it
does not rerun the visual encoder or alter any mapping artifact.  A fixed
hysteresis baseline is calibrated once on a designated sequence by matching
the adaptive detector's node count; room labels are reserved for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Protocol

import numpy as np

from vts_core.node_detection import AdaptiveValleyDetector, FixedValleyDetector

_COLD_FILENAME = re.compile(
    r"t\d+\.\d+_x-?\d+\.?\d*_y-?\d+\.?\d*_a-?\d+\.?\d*"
)
_COLD_CLASSES: tuple[str, ...] = (
    "CR",
    "2PO",
    "RL",
    "TL",
    "TR",
    "LO",
    "1PO",
    "KT",
    "CNR",
    "PA",
    "LAB",
    "ST",
)


class ValleyDetector(Protocol):
    """Minimal interface shared by the adaptive and fixed detectors."""

    def step(self, lambda_2: float) -> int | None:
        """Consume one sample and return a confirmed valley index, if any."""


@dataclass(frozen=True)
class SegmentationCase:
    """Inputs for one saved lambda_2 sequence."""

    name: str
    lambda2_path: str
    images_dir: str | None = None
    labels_path: str | None = None


@dataclass(frozen=True)
class Calibration:
    """Result of fitting the fixed baseline on one sequence."""

    case: str
    delta: float
    target_boundaries: int
    fixed_boundaries: int
    candidate_count: int


def _canonical_cold_label(raw: str) -> str:
    raw = raw.strip()
    for class_name in _COLD_CLASSES:
        if class_name in raw:
            return class_name
    return raw.split("-")[0].rstrip("0123456789")


def load_cold_frame_labels(images_dir: str, labels_path: str) -> list[str]:
    """Load COLD labels in exactly the image order used by the player."""
    image_root = Path(images_dir)
    annotation_path = Path(labels_path)
    if not image_root.is_dir():
        raise FileNotFoundError(f"COLD image directory not found: {images_dir}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COLD annotation file not found: {labels_path}")

    labels_by_name: dict[str, str] = {}
    for line in annotation_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            labels_by_name[os.path.basename(parts[0])] = _canonical_cold_label(
                parts[1]
            )

    image_names = sorted(
        path.name
        for path in image_root.iterdir()
        if path.is_file() and _COLD_FILENAME.search(path.name)
    )
    if not image_names:
        raise ValueError(f"No parsable COLD images found in {images_dir}")
    missing = [name for name in image_names if name not in labels_by_name]
    if missing:
        example = ", ".join(missing[:3])
        raise ValueError(
            f"Missing annotations for {len(missing)} images in {images_dir}: {example}"
        )
    return [labels_by_name[name] for name in image_names]


def detect_valleys(signal: np.ndarray, detector: ValleyDetector) -> list[int]:
    """Replay a streaming detector over a one-dimensional signal."""
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("lambda_2 signal must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(signal)):
        raise ValueError("lambda_2 signal contains non-finite values")

    valleys: list[int] = []
    for value in signal:
        valley = detector.step(float(value))
        if valley is not None:
            valleys.append(valley)
    return valleys


def valley_indices_to_splits(valleys: list[int], frame_count: int) -> list[int]:
    """Convert lambda_2 indices to exclusive frame-segment split positions."""
    # lambda_2[0] is produced by camera frame 1.  The mapper assigns the
    # valley frame to the preceding node, hence the following node starts at
    # valley_index + 2.
    return sorted({index + 2 for index in valleys if 0 < index + 2 < frame_count})


def _segments(splits: list[int], frame_count: int) -> list[tuple[int, int]]:
    edges = [0, *splits, frame_count]
    return list(pairwise(edges))


def _match_boundaries(
    detected: list[int], reference: list[int], tolerance_frames: int
) -> int:
    candidates = sorted(
        (
            abs(detected_position - reference_position),
            detected_index,
            reference_index,
        )
        for detected_index, detected_position in enumerate(detected)
        for reference_index, reference_position in enumerate(reference)
        if abs(detected_position - reference_position) <= tolerance_frames
    )
    used_detected: set[int] = set()
    used_reference: set[int] = set()
    for _, detected_index, reference_index in candidates:
        if detected_index in used_detected or reference_index in used_reference:
            continue
        used_detected.add(detected_index)
        used_reference.add(reference_index)
    return len(used_reference)


def _semantic_metrics(
    labels: list[str], splits: list[int], tolerance_frames: int
) -> dict[str, int | float]:
    transitions = [
        index
        for index in range(1, len(labels))
        if labels[index] != labels[index - 1]
    ]
    matches = _match_boundaries(splits, transitions, tolerance_frames)
    segments = _segments(splits, len(labels))

    dominant_frames = sum(
        max(Counter(labels[start:end]).values()) for start, end in segments
    )
    episode_edges = [0, *transitions, len(labels)]
    episodes = list(pairwise(episode_edges))
    segment_overlaps_per_episode = [
        sum(
            segment_start < end and start < segment_end
            for segment_start, segment_end in segments
        )
        for start, end in episodes
    ]
    episode_overlaps_per_segment = [
        sum(start < segment_end and segment_start < end for start, end in episodes)
        for segment_start, segment_end in segments
    ]

    return {
        "semantic_transitions": len(transitions),
        "matched_transitions": matches,
        "transition_recall": matches / len(transitions) if transitions else 1.0,
        "semantic_boundary_precision": matches / len(splits) if splits else 0.0,
        "frame_weighted_node_purity": dominant_frames / len(labels),
        "mean_segments_per_semantic_episode": float(
            np.mean(segment_overlaps_per_episode)
        ),
        "mean_semantic_episodes_per_segment": float(
            np.mean(episode_overlaps_per_segment)
        ),
    }


def segmentation_metrics(
    signal: np.ndarray,
    valleys: list[int],
    labels: list[str] | None,
    tolerance_frames: int,
    frame_rate_hz: float,
) -> dict[str, int | float | None]:
    """Compute granularity and optional semantic segmentation metrics."""
    frame_count = int(signal.size) + 1
    if labels is not None and len(labels) != frame_count:
        raise ValueError(
            f"Expected {frame_count} labels for the lambda_2 signal, got {len(labels)}"
        )
    splits = valley_indices_to_splits(valleys, frame_count)
    lengths = np.diff(np.asarray([0, *splits, frame_count], dtype=np.int64))
    result: dict[str, int | float | None] = {
        "frames": frame_count,
        "boundaries": len(splits),
        "nodes": len(splits) + 1,
        "nodes_per_1000_frames": (len(splits) + 1) * 1000.0 / frame_count,
        "median_segment_frames": float(np.median(lengths)),
        "median_segment_seconds": float(np.median(lengths) / frame_rate_hz),
        "minimum_segment_frames": int(np.min(lengths)),
        "maximum_segment_frames": int(np.max(lengths)),
        "semantic_transitions": None,
        "matched_transitions": None,
        "transition_recall": None,
        "semantic_boundary_precision": None,
        "frame_weighted_node_purity": None,
        "mean_segments_per_semantic_episode": None,
        "mean_semantic_episodes_per_segment": None,
    }
    if labels is not None:
        result.update(_semantic_metrics(labels, splits, tolerance_frames))
    return result


def calibrate_fixed_delta(
    signal: np.ndarray,
    target_boundaries: int,
    warmup: int,
    candidate_count: int = 4096,
) -> Calibration:
    """Fit a fixed delta by matching a target boundary count.

    All candidates with the smallest count error are retained and the one
    nearest their geometric center is selected. This prevents semantic labels
    from leaking into baseline calibration and avoids selecting an arbitrary
    interval edge.
    """
    if target_boundaries < 0:
        raise ValueError("target_boundaries cannot be negative")
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least 2")
    span = float(np.max(signal) - np.min(signal))
    upper = max(span, 1e-5)
    candidates = np.geomspace(1e-6, upper, candidate_count)
    counts = np.asarray(
        [
            len(detect_valleys(signal, FixedValleyDetector(float(delta), warmup)))
            for delta in candidates
        ],
        dtype=np.int64,
    )
    errors = np.abs(counts - target_boundaries)
    best = candidates[errors == np.min(errors)]
    geometric_center = float(np.exp(np.mean(np.log(best))))
    selected = float(best[np.argmin(np.abs(np.log(best / geometric_center)))])
    fixed_boundaries = len(
        detect_valleys(signal, FixedValleyDetector(selected, warmup))
    )
    return Calibration(
        case="",
        delta=selected,
        target_boundaries=target_boundaries,
        fixed_boundaries=fixed_boundaries,
        candidate_count=candidate_count,
    )


def _load_case(case: SegmentationCase) -> tuple[np.ndarray, list[str] | None]:
    if not os.path.isfile(case.lambda2_path):
        raise FileNotFoundError(f"lambda_2 trace not found: {case.lambda2_path}")
    signal = np.asarray(np.load(case.lambda2_path), dtype=np.float64)
    labels: list[str] | None = None
    if (case.images_dir is None) != (case.labels_path is None):
        raise ValueError(f"{case.name}: images and labels must be provided together")
    if case.images_dir is not None and case.labels_path is not None:
        labels = load_cold_frame_labels(case.images_dir, case.labels_path)
    return signal, labels


def run_experiment(
    cases: list[SegmentationCase],
    calibration_case: str,
    k_values: list[float],
    final_k: float,
    history: int,
    warmup: int,
    tolerance_frames: int,
    frame_rate_hz: float,
) -> tuple[Calibration, list[dict[str, object]]]:
    """Run the sensitivity sweep and calibrated fixed-baseline comparison."""
    if len({case.name for case in cases}) != len(cases):
        raise ValueError("case names must be unique")
    if calibration_case not in {case.name for case in cases}:
        raise ValueError(f"unknown calibration case: {calibration_case}")
    if not k_values or any(not math.isfinite(k) or k <= 0.0 for k in k_values):
        raise ValueError("k values must be positive and finite")
    if final_k not in k_values:
        raise ValueError("final k must be included in the sensitivity sweep")
    if tolerance_frames < 0:
        raise ValueError("tolerance_frames cannot be negative")
    if not math.isfinite(frame_rate_hz) or frame_rate_hz <= 0.0:
        raise ValueError("frame_rate_hz must be positive and finite")

    loaded = {case.name: _load_case(case) for case in cases}
    calibration_signal = loaded[calibration_case][0]
    target = len(
        detect_valleys(
            calibration_signal,
            AdaptiveValleyDetector(k=final_k, history=history, warmup=warmup),
        )
    )
    raw_calibration = calibrate_fixed_delta(
        calibration_signal, target, warmup=warmup
    )
    calibration = Calibration(
        case=calibration_case,
        delta=raw_calibration.delta,
        target_boundaries=raw_calibration.target_boundaries,
        fixed_boundaries=raw_calibration.fixed_boundaries,
        candidate_count=raw_calibration.candidate_count,
    )

    rows: list[dict[str, object]] = []
    for case in cases:
        signal, labels = loaded[case.name]
        for k in k_values:
            valleys = detect_valleys(
                signal,
                AdaptiveValleyDetector(k=k, history=history, warmup=warmup),
            )
            rows.append(
                {
                    "environment": case.name,
                    "method": "adaptive",
                    "parameter_name": "k",
                    "parameter_value": k,
                    **segmentation_metrics(
                        signal,
                        valleys,
                        labels,
                        tolerance_frames,
                        frame_rate_hz,
                    ),
                }
            )
        fixed_valleys = detect_valleys(
            signal, FixedValleyDetector(calibration.delta, warmup)
        )
        rows.append(
            {
                "environment": case.name,
                "method": "fixed",
                "parameter_name": "delta",
                "parameter_value": calibration.delta,
                **segmentation_metrics(
                    signal,
                    fixed_valleys,
                    labels,
                    tolerance_frames,
                    frame_rate_hz,
                ),
            }
        )
    return calibration, rows


def aggregate_results(
    rows: list[dict[str, object]], final_k: float
) -> dict[str, object]:
    """Summarize transfer behavior and the adaptive sensitivity sweep."""
    final_rows = [
        row
        for row in rows
        if row["method"] == "fixed"
        or (row["method"] == "adaptive" and row["parameter_value"] == final_k)
    ]
    comparison: dict[str, object] = {}
    for method in ("adaptive", "fixed"):
        selected = [row for row in final_rows if row["method"] == method]
        node_rates = np.asarray(
            [float(row["nodes_per_1000_frames"]) for row in selected]
        )
        labeled = [row for row in selected if row["transition_recall"] is not None]
        comparison[method] = {
            "total_nodes": sum(int(row["nodes"]) for row in selected),
            "mean_nodes_per_1000_frames": float(np.mean(node_rates)),
            "node_rate_coefficient_of_variation": float(
                np.std(node_rates) / np.mean(node_rates)
            ),
            "cold_macro_transition_recall": float(
                np.mean([float(row["transition_recall"]) for row in labeled])
            ),
            "cold_macro_frame_weighted_node_purity": float(
                np.mean(
                    [float(row["frame_weighted_node_purity"]) for row in labeled]
                )
            ),
            "cold_macro_segments_per_semantic_episode": float(
                np.mean(
                    [
                        float(row["mean_segments_per_semantic_episode"])
                        for row in labeled
                    ]
                )
            ),
        }

    sensitivity: list[dict[str, int | float]] = []
    k_values = sorted(
        {
            float(row["parameter_value"])
            for row in rows
            if row["method"] == "adaptive"
        }
    )
    for k in k_values:
        selected = [
            row
            for row in rows
            if row["method"] == "adaptive" and row["parameter_value"] == k
        ]
        labeled = [row for row in selected if row["transition_recall"] is not None]
        sensitivity.append(
            {
                "k": k,
                "total_nodes": sum(int(row["nodes"]) for row in selected),
                "mean_nodes_per_1000_frames": float(
                    np.mean(
                        [float(row["nodes_per_1000_frames"]) for row in selected]
                    )
                ),
                "cold_macro_transition_recall": float(
                    np.mean([float(row["transition_recall"]) for row in labeled])
                ),
                "cold_macro_frame_weighted_node_purity": float(
                    np.mean(
                        [
                            float(row["frame_weighted_node_purity"])
                            for row in labeled
                        ]
                    )
                ),
            }
        )
    return {"final_adaptive_vs_fixed": comparison, "adaptive_k_sweep": sensitivity}


def _write_csv(path: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(
    path: str,
    calibration: Calibration,
    rows: list[dict[str, object]],
    protocol: dict[str, object],
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(
            {
                "protocol": protocol,
                "fixed_baseline_calibration": asdict(calibration),
                "summary": aggregate_results(
                    rows, float(protocol["final_adaptive_k"])
                ),
                "results": rows,
            },
            handle,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        nargs=2,
        default=[],
        metavar=("NAME", "LAMBDA2"),
        help="Unlabeled sequence; repeat for multiple sequences",
    )
    parser.add_argument(
        "--labeled-case",
        action="append",
        nargs=4,
        default=[],
        metavar=("NAME", "LAMBDA2", "IMAGES", "LABELS"),
        help="COLD sequence with frame-level place annotations",
    )
    parser.add_argument("--calibration-case", required=True)
    parser.add_argument("--k-values", nargs="+", type=float, default=[1, 1.5, 2, 2.5, 3])
    parser.add_argument("--final-k", type=float, default=2.0)
    parser.add_argument("--history", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--transition-tolerance-frames", type=int, default=15)
    parser.add_argument("--frame-rate-hz", type=float, default=5.0)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    cases = [
        *(SegmentationCase(name, signal) for name, signal in args.case),
        *(
            SegmentationCase(name, signal, images, labels)
            for name, signal, images, labels in args.labeled_case
        ),
    ]
    if not cases:
        parser.error("at least one --case or --labeled-case is required")

    try:
        calibration, rows = run_experiment(
            cases=cases,
            calibration_case=args.calibration_case,
            k_values=args.k_values,
            final_k=args.final_k,
            history=args.history,
            warmup=args.warmup,
            tolerance_frames=args.transition_tolerance_frames,
            frame_rate_hz=args.frame_rate_hz,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    protocol: dict[str, object] = {
        "mapping_recomputed": False,
        "adaptive_k_values": args.k_values,
        "final_adaptive_k": args.final_k,
        "rolling_history_samples": args.history,
        "warmup_samples": args.warmup,
        "frame_rate_hz": args.frame_rate_hz,
        "transition_tolerance_frames": args.transition_tolerance_frames,
        "transition_tolerance_seconds": (
            args.transition_tolerance_frames / args.frame_rate_hz
        ),
        "fixed_delta_uses_semantic_labels": False,
        "fixed_delta_calibration_objective": (
            "match final adaptive boundary count on calibration case"
        ),
    }
    if args.output_json:
        _write_json(args.output_json, calibration, rows, protocol)
    if args.output_csv:
        _write_csv(args.output_csv, rows)
    if not args.output_json and not args.output_csv:
        json.dump(
            {
                "protocol": protocol,
                "fixed_baseline_calibration": asdict(calibration),
                "summary": aggregate_results(rows, args.final_k),
                "results": rows,
            },
            sys.stdout,
            indent=2,
        )
        print()
    destinations = [path for path in (args.output_json, args.output_csv) if path]
    if destinations:
        print(f"Node-segmentation experiment written to {', '.join(destinations)}")


if __name__ == "__main__":
    main()
