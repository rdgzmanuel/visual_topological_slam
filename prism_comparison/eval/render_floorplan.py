"""Render PRISM-TopoMap results on the COLD floorplans, VTS-figure style.

Uses the same calibration sidecars (TPS), floorplan images, room-label
canonicalization, and visual conventions as the VTS paper figures
(map_viz.render_on_floorplan): green GT trajectory, steelblue edges, nodes
coloured by canonical room label, abbreviated-label legend, no node ids.

Run on the host: python3 eval/render_floorplan.py [--results results]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
MAPS_DIR = REPO / "vts_devcontainer" / "vts_ws" / "images" / "maps"
FT_PATH = (REPO / "vts_devcontainer" / "vts_ws" / "src" / "vts_evaluation"
           / "vts_evaluation" / "floorplan_transform.py")
ENVS = ["freiburg_a", "freiburg_ext", "saarbruecken_a", "saarbruecken_ext"]

COLD_CLASSES = ("CR", "2PO", "RL", "TL", "TR", "LO", "1PO", "KT", "CNR",
                "PA", "LAB", "ST")


def canonical_label(raw: str) -> str:
    raw = raw.strip()
    for class_name in COLD_CLASSES:
        if class_name in raw:
            return class_name
    return raw.split("-")[0].rstrip("0123456789")


def load_floorplan_transform():
    spec = importlib.util.spec_from_file_location("floorplan_transform", FT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["floorplan_transform"] = mod
    spec.loader.exec_module(mod)
    return mod


def render_env(env: str, results_dir: Path, ft) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    res = results_dir / env
    graph = json.loads((res / "graph" / "graph.json").read_text())
    frames = [json.loads(l) for l in (res / "frames.jsonl").read_text().splitlines()]
    data = np.load(PKG / "data" / f"{env}.npz", allow_pickle=True)

    calib = json.loads((MAPS_DIR / f"{env}.calib.json").read_text())
    transform = ft.build_transform(calib)
    floor = np.array(Image.open(MAPS_DIR / f"{env}.png").convert("RGB"))

    vertices = graph["vertices"]
    adj = graph["edges"]
    positions = np.array([v["pose_for_visualization"][:2] for v in vertices])

    # Room label per vertex: the frame at which the vertex was created.
    creation_labels = [canonical_label(f["label"]) for f in frames if f["new_vertex"]]
    labels = (creation_labels + ["?"] * len(vertices))[: len(vertices)]

    edges = sorted({(min(u, v), max(u, v)) for u, lst in enumerate(adj)
                    for v, _ in lst})

    fig, ax = plt.subplots(figsize=(9, 13))
    ax.imshow(floor)

    gt_xy = data["scan_gt"][:, :2]
    gx, gy = transform(gt_xy[:, 0], gt_xy[:, 1])
    ax.plot(gx, gy, color="tab:green", lw=1.2, alpha=0.7,
            label="Ground-truth trajectory")

    cols, rows = transform(positions[:, 0], positions[:, 1])
    for u, v in edges:
        ax.plot([cols[u], cols[v]], [rows[u], rows[v]],
                color="steelblue", lw=1.3, zorder=4)

    unique = sorted(set(labels))
    cmap = plt.get_cmap("tab10" if len(unique) <= 10 else "tab20")
    color_of = {label: cmap(k % cmap.N) for k, label in enumerate(unique)}
    ax.scatter(cols, rows, c=[color_of[label] for label in labels],
               s=85, edgecolors="black", linewidths=0.5, zorder=5)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=color_of[label],
                          markeredgecolor="black", markersize=9, label=label)
               for label in unique]
    handles.append(plt.Line2D([0], [0], color="tab:green", lw=1.5,
                              label="Ground-truth trajectory"))
    ax.legend(handles=handles, loc="upper right", fontsize=11,
              framealpha=0.9, ncol=1)
    ax.set_axis_off()
    ax.set_title(" ")
    fig.tight_layout()
    fig.savefig(res / "map_on_floorplan.png", dpi=180, bbox_inches="tight")
    fig.savefig(res / "map_on_floorplan.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"{env}: {len(vertices)} vertices, {len(edges)} edges -> "
          f"{res / 'map_on_floorplan.pdf'}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results",
                        help="results dir name inside prism_comparison")
    args = parser.parse_args()
    ft = load_floorplan_transform()
    for env in ENVS:
        res = PKG / args.results / env
        if not (res / "graph" / "graph.json").exists():
            print(f"[skip] {env}")
            continue
        render_env(env, PKG / args.results, ft)


if __name__ == "__main__":
    main()
