"""Topological-map visualization for the offline evaluator.

Renders the generated map to a PNG (plus a ``.pgf`` twin for LaTeX) so map
quality is inspectable at a glance:

- Nodes are coloured by room class (abbreviated class labels in the legend;
  the report gives the abbreviation-to-full-name key). When
  per-node ground truth is available each node is *painted* at the
  ground-truth coordinates of the frame that created it. Ground truth is a
  paint-time input only: the mapper decides where and when to place nodes
  from odometry alone and never reads GT.
- Edges are drawn between nodes in a single uniform colour.
- The GT trajectory is drawn underneath for spatial context.

Every figure is additionally exported as ``.pgf`` (when the local TeX
toolchain allows) so it can be ``\\input`` directly in LaTeX/Overleaf, where
fonts and text sizes follow the document instead of being baked into a PNG.

matplotlib is imported lazily and the renderer degrades gracefully (returns
False with a warning) if it is unavailable, so it never breaks the metrics run.
"""

from __future__ import annotations

import json
import os

import numpy as np

from vts_core.topo_graph import TopoGraph

# Legend text sizes for the raster (PNG) preview. These only affect the PNG;
# the LaTeX-editable .pgf/.svg twin re-typesets all text in the document's font,
# so final publication sizing is controlled from Overleaf, not here.
_LEGEND_FONTSIZE: int = 16
_LEGEND_TITLE_FONTSIZE: int = 18
_LEGEND_MARKERSIZE: int = 11


def _save_latex_figure(fig, out_path: str) -> str | None:
    """Save a LaTeX-editable twin of the figure next to ``out_path``.

    Preferred format is ``.pgf`` (LaTeX source: ``\\input`` it in Overleaf and
    all text is typeset in the document's font, so size and placement stay
    editable after export), but writing PGF needs a local TeX toolchain for
    text metrics. Without one, fall back to ``.svg``: with Overleaf's ``svg``
    package (``\\usepackage{svg}`` + ``\\includesvg{...}``) the text layer is
    likewise re-typeset by LaTeX in the document's font.
    """
    pgf_path: str = os.path.splitext(out_path)[0] + ".pgf"
    try:
        fig.savefig(pgf_path, bbox_inches="tight")
        print(f"[viz] LaTeX (PGF) version saved to {pgf_path}")
        return pgf_path
    except Exception:  # no local TeX; SVG keeps the text editable in Overleaf
        svg_path: str = os.path.splitext(out_path)[0] + ".svg"
        try:
            fig.savefig(svg_path, bbox_inches="tight")
        except Exception as error:  # pragma: no cover - environment dependent
            print(f"[viz] LaTeX-editable export skipped ({error})")
            return None
        print(
            f"[viz] LaTeX-editable (SVG) version saved to {svg_path} "
            "(use \\usepackage{svg} + \\includesvg in Overleaf)"
        )
        return svg_path


def render_map(
    graph: TopoGraph,
    out_path: str,
    gt_xy: np.ndarray | None = None,
    node_gt: dict[int, np.ndarray] | None = None,
    title: str | None = None,
) -> bool:
    """Render the map to ``out_path`` (PNG + PGF). Returns True on success.

    When ``node_gt`` covers every node, nodes are painted at their
    ground-truth coordinates (paint-time only — the mapper placed them from
    odometry); otherwise the mapper's own (anchored-odometry) positions are
    drawn.
    """
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

    gt_pos: np.ndarray | None = None
    if node_gt is not None and all(i in node_gt for i in ids):
        gt_pos = np.array([node_gt[i] for i in ids], dtype=np.float64)
    drawn: np.ndarray = gt_pos if gt_pos is not None else map_pos

    fig, ax = plt.subplots(figsize=(11, 8))

    if gt_xy is not None and gt_xy.size:
        ax.plot(
            gt_xy[:, 0], gt_xy[:, 1], color="0.82", lw=1.0, zorder=1,
            label="Ground-truth trajectory",
        )

    for id_a, id_b in graph.edges():
        ka, kb = index[id_a], index[id_b]
        ax.plot(
            [drawn[ka, 0], drawn[kb, 0]],
            [drawn[ka, 1], drawn[kb, 1]],
            color="steelblue", lw=1.2, alpha=0.9, zorder=4,
        )

    labels: list[str] = [graph.nodes[i].room_label or "?" for i in ids]
    unique: list[str] = sorted(set(labels))
    cmap = plt.get_cmap("tab10" if len(unique) <= 10 else "tab20")
    color_of: dict[str, tuple] = {
        label: cmap(k % cmap.N) for k, label in enumerate(unique)
    }
    ax.scatter(
        drawn[:, 0], drawn[:, 1],
        c=[color_of[label] for label in labels],
        s=90, edgecolors="black", linewidths=0.5, zorder=5,
    )

    room_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color_of[label],
                   markeredgecolor="black", markersize=_LEGEND_MARKERSIZE, label=label)
        for label in unique
    ]
    edge_handles = [
        plt.Line2D([0], [0], color="steelblue", lw=2, label="Edge"),
    ]
    ax.legend(
        handles=room_handles + edge_handles, loc="best",
        fontsize=_LEGEND_FONTSIZE, title="Room",
        title_fontsize=_LEGEND_TITLE_FONTSIZE, framealpha=0.9, ncol=1,
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title or "Topological map")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    _save_latex_figure(fig, out_path)
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
    """Overlay the map on a metric floorplan image.

    Nodes are painted at their ground-truth coordinates (paint-time only —
    the mapper placed them from odometry), so per-node ground truth is
    required, along with a ``<floorplan>.calib.json`` sidecar giving the
    world-metres -> pixel affine. Degrades gracefully (returns False) if
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

    # Nodes are painted at their ground-truth coordinates, which live
    # directly in the floorplan's world frame (no alignment needed).
    gt_pos: np.ndarray = np.array([node_gt[i] for i in ids], dtype=np.float64)

    floor = np.array(Image.open(floorplan_path).convert("RGB"))
    height, width = floor.shape[:2]
    fig, ax = plt.subplots(figsize=(9, 13))
    ax.imshow(floor)

    if gt_xy is not None and gt_xy.size:
        gx, gy = transform(gt_xy[:, 0], gt_xy[:, 1])
        ax.plot(
            gx, gy, color="tab:green", lw=1.2, alpha=0.7,
            label="Ground-truth trajectory",
        )

    cols, rows = transform(gt_pos[:, 0], gt_pos[:, 1])
    index: dict[int, int] = {nid: k for k, nid in enumerate(ids)}
    for id_a, id_b in graph.edges():
        ka, kb = index[id_a], index[id_b]
        ax.plot(
            [cols[ka], cols[kb]], [rows[ka], rows[kb]],
            color="steelblue", lw=1.3, zorder=4,
        )

    labels: list[str] = [graph.nodes[i].room_label or "?" for i in ids]
    unique: list[str] = sorted(set(labels))
    cmap = plt.get_cmap("tab10" if len(unique) <= 10 else "tab20")
    color_of = {label: cmap(k % cmap.N) for k, label in enumerate(unique)}
    ax.scatter(
        cols, rows, c=[color_of[label] for label in labels],
        s=85, edgecolors="black", linewidths=0.5, zorder=5,
    )

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color_of[label],
                   markeredgecolor="black", markersize=_LEGEND_MARKERSIZE, label=label)
        for label in unique
    ]
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    ax.legend(
        handles=handles, loc="lower left", fontsize=_LEGEND_FONTSIZE,
        title="Room", title_fontsize=_LEGEND_TITLE_FONTSIZE,
    )
    ax.set_title(title or "Map on floorplan")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    _save_latex_figure(fig, out_path)
    plt.close(fig)
    print(f"[viz] floorplan overlay saved to {out_path}")
    return True
