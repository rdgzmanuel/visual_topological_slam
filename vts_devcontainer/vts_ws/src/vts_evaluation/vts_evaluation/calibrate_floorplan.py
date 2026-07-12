"""Fit a world-metres -> floorplan-pixel calibration for the map overlay.

The COLD pose coordinates are not 1:1 with the CAD floorplan's metric axes, so
a path drawn at face value collapses into the middle of the image. Because the
robot traverses the whole building, the fix is a **bounding-box fit**: stretch
the trajectory's extent to the building's extent. This is fully automatic and
is the recommended path for every environment:

    python -m vts_evaluation.calibrate_floorplan \
        --gt-trajectory <cold seq>/std_cam --floorplan images/maps/freiburg_a.png

It detects the building footprint (excluding the printed axes), maps the GT
trajectory's (x, y) extent onto it, writes a ``type: linear`` sidecar
``<floorplan>.calib.json`` consumed by the evaluator's ``--floorplan`` overlay,
and saves a ``<floorplan_stem>_calib_preview.png`` so you can sanity-check the
fit. Orientation is x -> vertical, y -> horizontal (COLD convention); pass
``--flip-x`` / ``--flip-y`` if a particular floorplan is mirrored, or
``--building-bbox r0 r1 c0 c1`` to override auto-detection.

For a *perfect* (warp-corrected) fit, supply explicit correspondences and fit a
thin-plate spline instead:

    python -m vts_evaluation.calibrate_floorplan --floorplan <png> \
        --correspondences pts.json   # {"world":[[x,y]..],"pixel":[[col,row]..]}
"""

from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

from vts_evaluation.floorplan_transform import build_transform

_COLD_FILENAME = re.compile(
    r"_x(?P<x>-?\d+\.?\d*)_y(?P<y>-?\d+\.?\d*)_a(?P<a>-?\d+\.?\d*)"
)


def gt_positions(images_dir: str) -> np.ndarray:
    """Parse all (x, y) ground-truth poses from a COLD image directory."""
    points: list[tuple[float, float]] = []
    for name in sorted(os.listdir(images_dir)):
        match = _COLD_FILENAME.search(name)
        if match is not None:
            points.append((float(match.group("x")), float(match.group("y"))))
    if not points:
        raise RuntimeError(f"No parsable COLD poses in {images_dir}")
    return np.array(points, dtype=np.float64)


def building_bbox(gray: np.ndarray, margin: int = 8) -> tuple[int, int, int, int]:
    """Detect the building footprint (rows/cols), excluding the printed axes.

    The CAD floorplans draw a vertical X-axis on the right and a horizontal
    Y-axis along the bottom, with tick labels beyond them. We find those two
    long lines and take the bounding box of the wall pixels inside them.
    """
    height, width = gray.shape
    dark = gray < 100
    x_axis_col = int(np.argmax(dark[:, int(width * 0.62):].sum(axis=0))) + int(
        width * 0.62
    )
    y_axis_row = int(np.argmax(dark[int(height * 0.62):, :].sum(axis=1))) + int(
        height * 0.62
    )
    region = dark.copy()
    region[:, max(x_axis_col - margin, 0):] = False
    region[max(y_axis_row - margin, 0):, :] = False
    rows, cols = np.where(region)
    if rows.size == 0:
        raise RuntimeError("Could not detect a building footprint")
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def bbox_fit_calibration(
    poses: np.ndarray,
    bbox: tuple[int, int, int, int],
    flip_x: bool = False,
    flip_y: bool = False,
    percentile: float = 0.0,
    swap_xy: bool = False,
) -> dict:
    """Linear calib mapping the trajectory's extent onto the building's extent.

    Args:
        poses: (N, 2) ground-truth (x, y).
        bbox: (row_min, row_max, col_min, col_max) building footprint.
        flip_x/flip_y: invert an axis if the floorplan is mirrored.
        percentile: trim this fraction (e.g. 0.01) off each extreme so a single
            stray pose cannot dictate the scale. 0 uses raw min/max.
        swap_xy: the floorplan is transposed (rotated 90 deg) — world x runs
            along the floorplan's columns and y along its rows, instead of the
            default x->rows, y->cols. Needed for floorplans drawn with the X
            axis horizontal (e.g. COLD Saarbruecken part A).

    Returns:
        A ``type: linear`` calibration dict. By default
        ``col = col_origin + col_per_y*y`` and ``row = row_origin + row_per_x*x``;
        with ``swap_xy`` the dependencies are transposed (``col_per_x`` /
        ``row_per_y``). ``flip_x``/``flip_y`` always refer to world x / world y.
    """
    r0, r1, c0, c1 = bbox
    lo, hi = percentile, 100.0 - percentile
    x_min, x_max = np.percentile(poses[:, 0], [lo, hi])
    y_min, y_max = np.percentile(poses[:, 1], [lo, hi])
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0

    def axis_fit(span_lo: int, span_hi: int, v_min: float, v_max: float,
                 v_span: float, flip: bool) -> tuple[float, float]:
        """(scale, origin) stretching world extent [v_min,v_max] onto the pixel
        span [span_lo, span_hi]. Default orientation puts larger world value at
        the smaller pixel index (up / left); ``flip`` reverses it."""
        if flip:
            scale = (span_hi - span_lo) / v_span
            origin = span_lo - scale * v_min
        else:
            scale = -(span_hi - span_lo) / v_span
            origin = span_lo - scale * v_max
        return scale, origin

    calib: dict = {
        "type": "linear",
        "_comment": (
            "Bounding-box fit: trajectory extent stretched to the building "
            "footprint (auto). Run calibrate_floorplan to regenerate."
        ),
    }
    if swap_xy:
        # x -> cols, y -> rows.
        col_per_x, col_origin = axis_fit(c0, c1, x_min, x_max, x_span, flip_x)
        row_per_y, row_origin = axis_fit(r0, r1, y_min, y_max, y_span, flip_y)
        calib.update(
            col_origin=float(col_origin), col_per_x=float(col_per_x),
            row_origin=float(row_origin), row_per_y=float(row_per_y),
        )
    else:
        # x -> rows, y -> cols (default).
        row_per_x, row_origin = axis_fit(r0, r1, x_min, x_max, x_span, flip_x)
        col_per_y, col_origin = axis_fit(c0, c1, y_min, y_max, y_span, flip_y)
        calib.update(
            col_origin=float(col_origin), col_per_y=float(col_per_y),
            row_origin=float(row_origin), row_per_x=float(row_per_x),
        )
    return calib


def _center_scale(calib: dict, scale: float, center_col: float, center_row: float) -> dict:
    """Fold a scale about (center_col, center_row) into a linear calib.

    Scales whichever per-axis coefficients are present, so it works for both
    the default (col_per_y / row_per_x) and the transposed (col_per_x /
    row_per_y) orientations.
    """
    out = dict(calib)
    for key in ("col_per_x", "col_per_y", "row_per_x", "row_per_y"):
        if key in calib:
            out[key] = scale * calib[key]
    out["col_origin"] = scale * calib["col_origin"] + center_col * (1.0 - scale)
    out["row_origin"] = scale * calib["row_origin"] + center_row * (1.0 - scale)
    return out


def wall_crossing_rate(
    gray: np.ndarray, poses: np.ndarray, calib: dict, near_px: float = 1.5
) -> float:
    """Fraction of trajectory points landing within ``near_px`` of a solid wall.

    The fit metric: lower is better. ~0 means the path stays inside corridors
    and rooms; a residual floor is unavoidable local warp (the COLD laser
    ground truth vs the CAD) plus legitimate doorway/furniture passes.
    """
    from scipy import ndimage

    height, width = gray.shape
    solid = gray < 70
    dist = ndimage.distance_transform_edt(~solid)
    col, row = build_transform(calib)(poses[:, 0], poses[:, 1])
    ci = np.clip(col, 0, width - 1).astype(int)
    ri = np.clip(row, 0, height - 1).astype(int)
    return float((dist[ri, ci] < near_px).mean())


def auto_scale(
    gray: np.ndarray,
    poses: np.ndarray,
    calib: dict,
    center: tuple[float, float],
    lo: float = 0.78,
    hi: float = 0.96,
    step: float = 0.02,
) -> float:
    """Pick the centred scale that minimises wall crossing in a gentle range.

    Restricted to a mild range so the path still fills the building (a wider
    search degenerates by shrinking the path to a tiny wall-free spot).
    """
    best_scale, best_cross = hi, 1.0
    for scale in np.round(np.arange(lo, hi + 1e-9, step), 3):
        scaled = _center_scale(calib, float(scale), center[0], center[1])
        cross = wall_crossing_rate(gray, poses, scaled)
        if cross < best_cross:
            best_cross, best_scale = cross, float(scale)
    return best_scale


def interactive_correct(
    floorplan_path: str,
    poses: np.ndarray,
    init_calib: dict,
    n_points: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Show the bbox-fit path with numbered markers; click each true location.

    Returns (world, pixel) correspondences for a TPS fit. Pixel-perfect: TPS
    passes through every clicked point and smoothly warps the rest.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    floor = np.array(Image.open(floorplan_path).convert("RGB"))
    height, width = floor.shape[:2]
    col, row = build_transform(init_calib)(poses[:, 0], poses[:, 1])
    marks = np.linspace(0, len(poses) - 1, n_points).astype(int)

    fig, ax = plt.subplots(figsize=(10, 13))
    ax.imshow(floor)
    ax.plot(col, row, color="lime", lw=0.8, alpha=0.5)
    for j, idx in enumerate(marks):
        ax.scatter([col[idx]], [row[idx]], c="red", s=60, zorder=5)
        ax.annotate(str(j + 1), (col[idx], row[idx]), color="red", fontsize=11, zorder=6)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    world: list = []
    pixel: list = []
    for j, idx in enumerate(marks):
        ax.set_title(
            f"Click the TRUE location of point {j + 1}/{n_points} "
            f"(red dot {j + 1} shows where it is now). Right-click to skip."
        )
        fig.canvas.draw()
        pts = plt.ginput(1, timeout=0, mouse_pop=3)
        if not pts:
            continue
        world.append(poses[idx])
        pixel.append(pts[0])
        ax.scatter([pts[0][0]], [pts[0][1]], c="blue", s=60, zorder=7)
        ax.annotate(str(j + 1), pts[0], color="blue", fontsize=11, zorder=8)
    plt.close(fig)
    return np.array(world, dtype=float), np.array(pixel, dtype=float)


def fit_and_save(
    world: np.ndarray, pixel: np.ndarray, out_path: str, kind: str
) -> dict:
    """Fit an affine/TPS from explicit correspondences and save the calib."""
    calib: dict = {
        "type": kind,
        "world": np.asarray(world, dtype=float).tolist(),
        "pixel": np.asarray(pixel, dtype=float).tolist(),
    }
    transform = build_transform(calib)
    pc, pr = transform(world[:, 0], world[:, 1])
    residual = np.hypot(pc - pixel[:, 0], pr - pixel[:, 1])
    calib["_residual_px"] = {
        "mean": round(float(residual.mean()), 3),
        "max": round(float(residual.max()), 3),
    }
    with open(out_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(
        f"[calib] {kind} from {len(world)} correspondences; residual "
        f"mean={calib['_residual_px']['mean']}px max={calib['_residual_px']['max']}px"
    )
    return calib


def _save_preview(
    floorplan_path: str, poses: np.ndarray, calib: dict, out_path: str
) -> None:
    """Render the GT path through ``calib`` over the floorplan for a sanity check."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except Exception as error:  # pragma: no cover
        print(f"[calib] no matplotlib, skipping preview: {error}")
        return
    floor = np.array(Image.open(floorplan_path).convert("RGB"))
    height, width = floor.shape[:2]
    col, row = build_transform(calib)(poses[:, 0], poses[:, 1])
    fig, ax = plt.subplots(figsize=(9, 13))
    ax.imshow(floor)
    ax.plot(col, row, color="lime", lw=1.0, alpha=0.8)
    ax.scatter([col[0]], [row[0]], c="red", s=150, marker="*", zorder=6)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.set_title("Calibration preview (GT path over floorplan)")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[calib] preview saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floorplan", required=True)
    parser.add_argument("--gt-trajectory", default="", help="COLD std_cam dir")
    parser.add_argument(
        "--correspondences",
        default="",
        help='JSON {"world":[[x,y]..],"pixel":[[col,row]..]} for a TPS/affine fit',
    )
    parser.add_argument("--kind", default="tps", choices=["tps", "affine"])
    parser.add_argument("--out", default="", help="default <floorplan>.calib.json")
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument(
        "--swap-xy", action="store_true",
        help="floorplan is transposed (X axis drawn horizontal): map world x "
             "to columns and y to rows. Needed for COLD Saarbruecken part A.",
    )
    parser.add_argument("--trim-percentile", type=float, default=0.0)
    parser.add_argument(
        "--scale", type=float, default=None,
        help="centred scale on top of the bbox fit (default: auto, ~0.8-0.9, "
             "shrinks the path off the outer walls). Tune while watching the "
             "reported wall-crossing rate.",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="pixel-perfect: click the true location of each numbered point on "
             "the bbox-fit path; fits a thin-plate spline through them.",
    )
    parser.add_argument("--n-points", type=int, default=10)
    parser.add_argument(
        "--building-bbox", nargs=4, type=int, default=None,
        metavar=("R0", "R1", "C0", "C1"), help="override auto footprint detection",
    )
    args = parser.parse_args()
    out_path = args.out or (os.path.splitext(args.floorplan)[0] + ".calib.json")

    if args.correspondences:
        with open(args.correspondences) as f:
            data = json.load(f)
        calib = fit_and_save(
            np.array(data["world"], float), np.array(data["pixel"], float),
            out_path, args.kind,
        )
        poses = np.array(data["world"], float)
    else:
        if not args.gt_trajectory:
            parser.error("need --gt-trajectory or --correspondences")
        from PIL import Image

        poses = gt_positions(args.gt_trajectory)
        gray = np.array(Image.open(args.floorplan).convert("L"))
        bbox = tuple(args.building_bbox) if args.building_bbox else building_bbox(gray)
        print(f"[calib] building footprint rows/cols: {bbox}")
        calib = bbox_fit_calibration(
            poses, bbox, args.flip_x, args.flip_y, args.trim_percentile,
            swap_xy=args.swap_xy,
        )
        r0, r1, c0, c1 = bbox
        center = ((c0 + c1) / 2.0, (r0 + r1) / 2.0)
        scale = args.scale if args.scale is not None else auto_scale(
            gray, poses, calib, center
        )
        calib = _center_scale(calib, scale, center[0], center[1])

        if args.interactive:
            world, pixel = interactive_correct(
                args.floorplan, poses, calib, args.n_points
            )
            if len(world) < 4:
                parser.error("need >= 4 clicked points for a TPS fit")
            calib = fit_and_save(world, pixel, out_path, "tps")
        else:
            crossing = wall_crossing_rate(gray, poses, calib)
            calib["_scale"] = round(float(scale), 3)
            calib["_wall_crossing_rate"] = round(crossing, 4)
            with open(out_path, "w") as f:
                json.dump(calib, f, indent=2)
            print(
                f"[calib] bbox fit, centred scale={scale:.2f}, "
                f"solid-wall crossing={crossing:.3f} -> {out_path}"
            )

    preview = os.path.splitext(args.floorplan)[0] + "_calib_preview.png"
    _save_preview(args.floorplan, poses, calib, preview)


if __name__ == "__main__":
    main()
