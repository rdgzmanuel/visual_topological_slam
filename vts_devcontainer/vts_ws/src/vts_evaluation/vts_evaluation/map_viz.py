"""Topological-map visualization for the offline evaluator.

Renders the generated map to a PNG so map quality is inspectable at a glance:

- Nodes are coloured by room label and annotated with their id.
- Edges are drawn between nodes; an edge that is *short in the map but long in
  ground truth* (a perceptual-aliasing false closure, the same rule as
  ``false_merge_rate``) is highlighted in red.
- When per-node ground truth is available the map is rigidly aligned to the GT
  frame (the same best-fit SE(2) used for placement RMSE), the GT trajectory
  is drawn underneath, and a thin red stem connects each node to its true
  position — so accumulated drift is visible directly.

matplotlib is imported lazily and the renderer degrades gracefully (returns
False with a warning) if it is unavailable, so it never breaks the metrics run.
"""

from __future__ import annotations

import json
import os

import numpy as np

from vts_core.matching import _fit_se2, median_spacing
from vts_core.topo_graph import TopoGraph


def render_map(
    graph: TopoGraph,
    out_path: str,
    gt_xy: np.ndarray | None = None,
    node_gt: dict[int, np.ndarray] | None = None,
    title: str | None = None,
) -> bool:
    """Render the map to ``out_path`` (PNG). Returns True on success."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover - environment dependent
        print(f"[viz] matplotlib unavailable, skipping map image: {error}")
        return False

    ids: list[int] = sorted(graph.nodes)
    if not ids:
        print("[viz] empty graph, nothing to draw")
        return False
    index: dict[int, int] = {nid: k for k, nid in enumerate(ids)}
    map_pos: np.ndarray = graph.positions()

    # Align the map to the GT frame when per-node GT is available.
    gt_pos: np.ndarray | None = None
    aligned: np.ndarray = map_pos
    if node_gt is not None and all(i in node_gt for i in ids) and len(ids) >= 2:
        gt_pos = np.array([node_gt[i] for i in ids], dtype=np.float64)
        fit = _fit_se2(map_pos, gt_pos)
        if fit is not None:
            rotation, translation = fit
            aligned = map_pos @ rotation.T + translation

    tolerance: float = median_spacing(gt_pos if gt_pos is not None else map_pos)

    fig, ax = plt.subplots(figsize=(11, 8))

    if gt_xy is not None and gt_xy.size:
        ax.plot(
            gt_xy[:, 0], gt_xy[:, 1], color="0.82", lw=1.0, zorder=1,
            label="GT trajectory",
        )
    if gt_pos is not None:
        for k in range(len(ids)):
            ax.plot(
                [aligned[k, 0], gt_pos[k, 0]], [aligned[k, 1], gt_pos[k, 1]],
                color="0.55", lw=0.7, alpha=0.6, linestyle=":", zorder=2,
            )
        ax.scatter(
            gt_pos[:, 0], gt_pos[:, 1], marker="x", c="0.4", s=22, lw=0.8,
            zorder=3, label="GT node position (stem = drift)",
        )

    n_false: int = 0
    for id_a, id_b in graph.edges():
        ka, kb = index[id_a], index[id_b]
        suspicious: bool = False
        if gt_pos is not None:
            gt_dist: float = float(np.linalg.norm(gt_pos[ka] - gt_pos[kb]))
            map_dist: float = float(np.linalg.norm(map_pos[ka] - map_pos[kb]))
            suspicious = gt_dist > map_dist + 3.0 * tolerance
        n_false += int(suspicious)
        ax.plot(
            [aligned[ka, 0], aligned[kb, 0]],
            [aligned[ka, 1], aligned[kb, 1]],
            color="crimson" if suspicious else "steelblue",
            lw=2.2 if suspicious else 1.2,
            alpha=0.9, zorder=4,
        )

    labels: list[str] = [graph.nodes[i].room_label or "?" for i in ids]
    unique: list[str] = sorted(set(labels))
    cmap = plt.get_cmap("tab10" if len(unique) <= 10 else "tab20")
    color_of: dict[str, tuple] = {
        label: cmap(k % cmap.N) for k, label in enumerate(unique)
    }
    ax.scatter(
        aligned[:, 0], aligned[:, 1],
        c=[color_of[label] for label in labels],
        s=90, edgecolors="black", linewidths=0.5, zorder=5,
    )
    for k, nid in enumerate(ids):
        ax.annotate(
            str(nid), (aligned[k, 0], aligned[k, 1]),
            fontsize=6, ha="center", va="center", zorder=6,
        )

    room_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color_of[label],
                   markeredgecolor="black", markersize=8, label=label)
        for label in unique
    ]
    edge_handles = [
        plt.Line2D([0], [0], color="steelblue", lw=2, label="edge"),
    ]
    if n_false:
        edge_handles.append(
            plt.Line2D([0], [0], color="crimson", lw=2, label="false-merge edge")
        )
    ax.legend(
        handles=room_handles + edge_handles, loc="best", fontsize=7,
        title="room", framealpha=0.9, ncol=1,
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title or "Topological map")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] map image saved to {out_path}")
    return True


def render_confusion(
    confusion: dict[str, dict[str, int]], out_path: str, title: str | None = None
) -> bool:
    """Render a row-normalized retrieval confusion matrix (true vs top-1)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover
        print(f"[viz] matplotlib unavailable, skipping confusion matrix: {error}")
        return False

    true_labels: list[str] = sorted(confusion)
    pred_labels: list[str] = sorted(
        {p for row in confusion.values() for p in row} | set(true_labels)
    )
    matrix: np.ndarray = np.zeros((len(true_labels), len(pred_labels)), dtype=float)
    for i, t in enumerate(true_labels):
        total = sum(confusion[t].values()) or 1
        for j, p in enumerate(pred_labels):
            matrix[i, j] = confusion[t].get(p, 0) / total

    fig, ax = plt.subplots(
        figsize=(max(6, len(pred_labels)), max(5, len(true_labels)))
    )
    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(pred_labels)), pred_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(true_labels)), true_labels)
    ax.set_xlabel("predicted (top-1) room")
    ax.set_ylabel("true room")
    ax.set_title(title or "Retrieval confusion (row-normalized)")
    for i in range(len(true_labels)):
        for j in range(len(pred_labels)):
            if matrix[i, j] > 0:
                ax.text(
                    j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] > 0.5 else "black", fontsize=8,
                )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] confusion matrix saved to {out_path}")
    return True


def render_rejection_curve(
    curve: list[dict[str, float]], out_path: str, title: str | None = None
) -> bool:
    """Plot the coverage-vs-precision operating curve of calibrated rejection."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover
        print(f"[viz] matplotlib unavailable, skipping rejection curve: {error}")
        return False
    if not curve:
        return False
    coverage = [p["coverage"] for p in curve]
    precision = [p["precision_at_1"] for p in curve]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverage, precision, "-o", color="tab:blue", markersize=3)
    ax.set_xlabel("coverage (fraction of queries answered)")
    ax.set_ylabel("precision@1 on answered queries")
    ax.set_title(title or "Calibrated rejection: coverage vs precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(min(0.5, min(precision) - 0.05), 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] rejection curve saved to {out_path}")
    return True


def _load_calibration(floorplan_path: str) -> dict[str, float] | None:
    """Load the ``<floorplan>.calib.json`` world-metres -> pixel affine."""
    stem: str = os.path.splitext(floorplan_path)[0]
    calib_path: str = stem + ".calib.json"
    if not os.path.exists(calib_path):
        print(f"[viz] no calibration sidecar at {calib_path}; skipping overlay")
        return None
    with open(calib_path) as f:
        calib: dict = json.load(f)
    return calib


def render_on_floorplan(
    graph: TopoGraph,
    floorplan_path: str,
    out_path: str,
    node_gt: dict[int, np.ndarray],
    gt_xy: np.ndarray | None = None,
    title: str | None = None,
) -> bool:
    """Overlay the map (aligned to ground truth) on a metric floorplan image.

    Requires per-node ground truth (to align the odometry-frame map into the
    floorplan's world frame) and a ``<floorplan>.calib.json`` sidecar giving
    the world-metres -> pixel affine. Degrades gracefully (returns False) if
    matplotlib, the floorplan, the calibration, or the GT are unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except Exception as error:  # pragma: no cover - environment dependent
        print(f"[viz] matplotlib/PIL unavailable, skipping floorplan: {error}")
        return False

    if not os.path.exists(floorplan_path):
        print(f"[viz] floorplan not found: {floorplan_path}")
        return False
    calib: dict[str, float] | None = _load_calibration(floorplan_path)
    if calib is None:
        return False
    ids: list[int] = sorted(graph.nodes)
    if not ids or not all(i in node_gt for i in ids):
        print("[viz] floorplan overlay needs per-node ground truth; skipping")
        return False

    from vts_evaluation.floorplan_transform import build_transform

    try:
        transform = build_transform(calib)
    except (KeyError, ValueError) as error:
        print(f"[viz] bad floorplan calibration: {error}")
        return False

    # Align the odometry-frame map into the world (floorplan) frame.
    map_pos: np.ndarray = graph.positions()
    gt_pos: np.ndarray = np.array([node_gt[i] for i in ids], dtype=np.float64)
    fit = _fit_se2(map_pos, gt_pos)
    aligned: np.ndarray = map_pos @ fit[0].T + fit[1] if fit else map_pos

    floor = np.array(Image.open(floorplan_path).convert("RGB"))
    height, width = floor.shape[:2]
    fig, ax = plt.subplots(figsize=(9, 13))
    ax.imshow(floor)

    if gt_xy is not None and gt_xy.size:
        gx, gy = transform(gt_xy[:, 0], gt_xy[:, 1])
        ax.plot(gx, gy, color="tab:green", lw=1.2, alpha=0.7, label="GT path")

    cols, rows = transform(aligned[:, 0], aligned[:, 1])
    index: dict[int, int] = {nid: k for k, nid in enumerate(ids)}
    tolerance: float = median_spacing(gt_pos)
    for id_a, id_b in graph.edges():
        ka, kb = index[id_a], index[id_b]
        suspicious: bool = float(
            np.linalg.norm(gt_pos[ka] - gt_pos[kb])
        ) > float(np.linalg.norm(map_pos[ka] - map_pos[kb])) + 3.0 * tolerance
        ax.plot(
            [cols[ka], cols[kb]], [rows[ka], rows[kb]],
            color="crimson" if suspicious else "steelblue",
            lw=2.0 if suspicious else 1.3, zorder=4,
        )

    labels: list[str] = [graph.nodes[i].room_label or "?" for i in ids]
    unique: list[str] = sorted(set(labels))
    cmap = plt.get_cmap("tab10" if len(unique) <= 10 else "tab20")
    color_of = {label: cmap(k % cmap.N) for k, label in enumerate(unique)}
    ax.scatter(
        cols, rows, c=[color_of[label] for label in labels],
        s=85, edgecolors="black", linewidths=0.5, zorder=5,
    )
    for k, nid in enumerate(ids):
        ax.annotate(
            str(nid), (cols[k], rows[k]), fontsize=6,
            ha="center", va="center", zorder=6,
        )

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color_of[label],
                   markeredgecolor="black", markersize=8, label=label)
        for label in unique
    ]
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.legend(handles=handles, loc="lower left", fontsize=7, title="room")
    ax.set_title(title or "Map on floorplan")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] floorplan overlay saved to {out_path}")
    return True
