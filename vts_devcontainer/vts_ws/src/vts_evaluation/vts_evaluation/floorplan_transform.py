"""World-metres -> floorplan-pixel transforms for the map overlay.

Three transform types, selected by the ``type`` field of a
``<floorplan>.calib.json`` sidecar:

- ``linear``  : per-axis affine read straight from the floorplan's printed
  axis ticks (``col = col_origin + col_per_y * y``;
  ``row = row_origin + row_per_x * x``). Correct *for the drawn axes*, but the
  COLD laser ground truth is itself smoothly warped relative to the CAD, so a
  linear map leaves visible residuals.
- ``affine`` : a full 6-DoF affine fit from >= 3 correspondences (handles
  rotation/shear/anisotropic scale that the axis-aligned linear map cannot).
- ``tps``    : a thin-plate spline fit from >= 4 correspondences. TPS passes
  through every control point exactly, is globally smooth and minimises
  bending energy — the principled replacement for the hand-fitted high-degree
  polynomial (which oscillates between and outside its control points).

All transforms expose the same call signature ``f(x, y) -> (col, row)`` on
NumPy arrays, so the renderer is agnostic to which one a floorplan uses.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Transform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------#
# Thin-plate spline
# ---------------------------------------------------------------------------#
def _tps_kernel(squared_distance: np.ndarray) -> np.ndarray:
    """Radial basis U(r) = r^2 log r, written in terms of r^2 (= 0.5 r^2 log r^2)."""
    out: np.ndarray = np.zeros_like(squared_distance)
    positive: np.ndarray = squared_distance > 1e-12
    out[positive] = 0.5 * squared_distance[positive] * np.log(
        squared_distance[positive]
    )
    return out


def fit_tps(
    src: np.ndarray, dst: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a thin-plate spline mapping src (N,2) world -> dst (N,2) pixels.

    Returns:
        (control_points, weights) where weights has shape (N+3, 2): the first N
        rows are the RBF weights, the last 3 the affine part.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n: int = src.shape[0]
    if n < 4:
        raise ValueError("TPS needs at least 4 correspondences")
    diff: np.ndarray = src[:, None, :] - src[None, :, :]
    squared: np.ndarray = np.einsum("ijk,ijk->ij", diff, diff)
    k_mat: np.ndarray = _tps_kernel(squared)
    p_mat: np.ndarray = np.column_stack([np.ones(n), src])  # (N,3)

    upper: np.ndarray = np.hstack([k_mat, p_mat])
    lower: np.ndarray = np.hstack([p_mat.T, np.zeros((3, 3))])
    l_mat: np.ndarray = np.vstack([upper, lower])  # (N+3, N+3)
    target: np.ndarray = np.vstack([dst, np.zeros((3, 2))])  # (N+3, 2)

    weights: np.ndarray = np.linalg.solve(
        l_mat + np.eye(n + 3) * 1e-9, target
    )
    return src, weights


def apply_tps(
    control: np.ndarray, weights: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a fitted TPS to query points."""
    query: np.ndarray = np.column_stack([np.ravel(x), np.ravel(y)])
    diff: np.ndarray = query[:, None, :] - control[None, :, :]
    squared: np.ndarray = np.einsum("ijk,ijk->ij", diff, diff)
    u_mat: np.ndarray = _tps_kernel(squared)
    n: int = control.shape[0]
    affine_part: np.ndarray = np.column_stack(
        [np.ones(query.shape[0]), query]
    ) @ weights[n:]
    out: np.ndarray = u_mat @ weights[:n] + affine_part
    return out[:, 0], out[:, 1]


# ---------------------------------------------------------------------------#
# Affine
# ---------------------------------------------------------------------------#
def fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 6-DoF affine [a b c; d e f] mapping world -> pixel."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape[0] < 3:
        raise ValueError("Affine needs at least 3 correspondences")
    design: np.ndarray = np.column_stack([src, np.ones(src.shape[0])])  # (N,3)
    solution, *_ = np.linalg.lstsq(design, dst, rcond=None)  # (3,2)
    return solution.T  # (2,3): rows = [col; row]


# ---------------------------------------------------------------------------#
# Dispatch
# ---------------------------------------------------------------------------#
def build_transform(calib: dict) -> Transform:
    """Build a world->pixel transform callable from a calibration dict."""
    kind: str = str(calib.get("type", "linear"))

    if kind == "linear":
        # Per-axis affine. Each pixel axis may depend on either world axis, so a
        # transposed/rotated floorplan (x->cols, y->rows) is expressible via the
        # cross terms. Missing coefficients default to 0, keeping older sidecars
        # (which only set col_per_y / row_per_x) working unchanged.
        c0 = float(calib["col_origin"])
        r0 = float(calib["row_origin"])
        cx = float(calib.get("col_per_x", 0.0))
        cy = float(calib.get("col_per_y", 0.0))
        rx = float(calib.get("row_per_x", 0.0))
        ry = float(calib.get("row_per_y", 0.0))

        def linear(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            x_arr, y_arr = np.asarray(x), np.asarray(y)
            return c0 + cx * x_arr + cy * y_arr, r0 + rx * x_arr + ry * y_arr

        return linear

    src: np.ndarray = np.asarray(calib["world"], dtype=np.float64)
    dst: np.ndarray = np.asarray(calib["pixel"], dtype=np.float64)

    if kind == "affine":
        matrix: np.ndarray = fit_affine(src, dst)

        def affine(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            pts = np.column_stack([np.ravel(x), np.ravel(y), np.ones(np.size(x))])
            out = pts @ matrix.T
            return out[:, 0], out[:, 1]

        return affine

    if kind == "tps":
        control, weights = fit_tps(src, dst)

        def tps(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return apply_tps(control, weights, x, y)

        return tps

    raise ValueError(f"Unknown floorplan transform type: {kind!r}")
