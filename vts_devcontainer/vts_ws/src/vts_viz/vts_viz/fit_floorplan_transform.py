"""Floorplan visualization utilities (LEGACY similarity-only fitter).

SUPERSEDED by ``vts_evaluation.calibrate_floorplan``, which fits a thin-plate
spline from room-centroid correspondences and is consumed directly by the
evaluator's ``--floorplan`` overlay. A rigid 4-DoF similarity (this file) is
too stiff for the COLD ground-truth-vs-CAD warp; the TPS captures it without
the oscillation of the original 5th-degree polynomial. Kept only for
reference.

The thesis used hand-fitted 5th-degree polynomial weights per environment to
warp odometry coordinates onto a floorplan PNG. Visualization is the one
legitimately dataset-specific component, but a 12-coefficient polynomial is
neither interpretable nor reusable. It is replaced here by a 2-D similarity
transform (rotation + uniform scale + translation, 4 DoF) fit by least
squares from a handful of correspondences — clicked once per floorplan and
stored in the environment's YAML. Run:

    python -m vts_viz.fit_floorplan_transform \
        --gt-trajectory <cold images dir> --floorplan <png> --out <yaml>

and click >= 3 trajectory landmarks on the floorplan when prompted. The
overlay function then draws any TopoGraph on the floorplan with that
transform. None of this touches the mapping pipeline.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import yaml

from vts_core.topo_graph import TopoGraph


def fit_similarity(world: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform world (m) -> floorplan (px).

    Args:
        world: (N, 2) world coordinates of landmarks.
        pixels: (N, 2) corresponding pixel coordinates.

    Returns:
        2x3 affine matrix [sR | t] usable with cv2.transform.
    """
    if world.shape[0] < 3:
        raise ValueError("Need at least 3 correspondences")
    matrix, _ = cv2.estimateAffinePartial2D(
        world.astype(np.float32).reshape(-1, 1, 2),
        pixels.astype(np.float32).reshape(-1, 1, 2),
        method=cv2.LMEDS,
    )
    if matrix is None:
        raise RuntimeError("Similarity fit failed")
    return matrix.astype(np.float64)


def overlay_graph(
    graph: TopoGraph, floorplan: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    """Draw graph nodes (red) and edges (green) on a floorplan image."""
    canvas: np.ndarray = floorplan.copy()
    ids: list[int] = sorted(graph.nodes)
    world: np.ndarray = np.array(
        [graph.nodes[i].pose[:2] for i in ids], dtype=np.float32
    ).reshape(-1, 1, 2)
    pixels: np.ndarray = cv2.transform(world, transform).reshape(-1, 2)
    pixel_of: dict[int, tuple[int, int]] = {
        node_id: (int(p[0]), int(p[1])) for node_id, p in zip(ids, pixels)
    }
    for id_a, id_b in graph.edges():
        cv2.line(canvas, pixel_of[id_a], pixel_of[id_b], (0, 255, 0), 2)
    for node_id in ids:
        cv2.circle(canvas, pixel_of[node_id], 5, (0, 0, 255), -1)
    return canvas


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-trajectory", required=True)
    parser.add_argument("--floorplan", required=True)
    parser.add_argument("--out", required=True, help="YAML to store the transform")
    args: argparse.Namespace = parser.parse_args()

    from vts_players.cold_player_node import parse_cold_filename
    import os

    points: list[tuple[float, float]] = []
    for name in sorted(os.listdir(args.gt_trajectory)):
        parsed = parse_cold_filename(name)
        if parsed is not None:
            points.append((parsed[1][0], parsed[1][1]))
    world: np.ndarray = np.array(points, dtype=np.float64)

    floorplan: np.ndarray = cv2.imread(args.floorplan)
    if floorplan is None:
        raise FileNotFoundError(args.floorplan)

    # Interactive landmark clicking: the first/median/last trajectory points
    # are highlighted in the console; click their floorplan locations in order.
    landmark_indices: list[int] = [0, len(world) // 2, len(world) - 1]
    print("Click, in order, the floorplan locations of these world points:")
    for index in landmark_indices:
        print(f"  world ({world[index][0]:.2f}, {world[index][1]:.2f})")

    clicked: list[tuple[int, int]] = []

    def on_click(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x, y))
            print(f"  clicked ({x}, {y})")

    cv2.namedWindow("floorplan")
    cv2.setMouseCallback("floorplan", on_click)
    while len(clicked) < len(landmark_indices):
        cv2.imshow("floorplan", floorplan)
        if cv2.waitKey(30) == 27:
            raise SystemExit("Aborted")
    cv2.destroyAllWindows()

    transform: np.ndarray = fit_similarity(
        world[landmark_indices], np.array(clicked, dtype=np.float64)
    )
    with open(args.out, "w") as f:
        yaml.safe_dump({"floorplan_transform": transform.tolist()}, f)
    print(f"Saved transform to {args.out}")


if __name__ == "__main__":
    main()
