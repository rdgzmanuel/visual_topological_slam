"""Build the topological map WITH and WITHOUT odometry drift, side by side.

Extracts the place descriptors once, then runs the exact mapping pipeline twice
on the same descriptors — once with perfect odometry (alpha = 0, the map the
robot *would* build with no drift) and once with the configured odometry noise
— and renders both node graphs over the ground-truth path. Shows directly how
the drift scatters/duplicates nodes and inflates placement error.

    python3 -m vts_evaluation.compare_odometry_maps \
        --gt-trajectory /workspace/encoder/seq_data/<seq>/std_cam \
        --extractor finetuned:src \
        --model-name visual_encoder_dino_contrastive_dim128_best \
        --encoder-path /workspace/encoder \
        --alpha 0.025 0.005 0.01 0.0025 --valley-k 1.5 --odom-seed 17 \
        --cache /tmp/<seq>_desc.npz \
        --out output/freiburg_a/images/map_compare.png

``--cache`` stores the descriptors so re-running with different ``--alpha`` /
``--valley-k`` is instant (extraction is the only slow step). Needs torch + the
encoder (run it in the container).
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np

from vts_core.mapper import TopologicalMapper
from vts_core.matching import _fit_se2
from vts_core.metrics import graph_metrics
from vts_core.motion import OdometryNoiseParams, OdometrySimulator
from vts_core.topo_graph import TopoGraph

_POSE = re.compile(r"_x(?P<x>-?\d+\.?\d*)_y(?P<y>-?\d+\.?\d*)_a(?P<a>-?\d+\.?\d*)")


def _load_descriptors(
    images_dir: str, extractor_spec: str, model_name: str,
    encoder_path: str, cache: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (descriptors (N,d), poses (N,3)); cache to disk if requested."""
    if cache and os.path.exists(cache):
        data = np.load(cache)
        print(f"[compare] loaded cached descriptors from {cache}")
        return data["descs"], data["poses"]

    import cv2

    from vts_core.features import build_extractor

    names = [n for n in sorted(os.listdir(images_dir)) if _POSE.search(n)]
    poses = np.array(
        [[float(_POSE.search(n).group(g)) for g in ("x", "y", "a")] for n in names]
    )
    extractor = build_extractor(extractor_spec, model_name, encoder_path)
    descs: list[np.ndarray] = []
    for i, name in enumerate(names):
        image = cv2.imread(os.path.join(images_dir, name), cv2.IMREAD_COLOR)
        descs.append(extractor.extract(image))
        if i % 250 == 0:
            print(f"[compare] extracted {i}/{len(names)}", flush=True)
    descriptors = np.array(descs, dtype=np.float32)
    if cache:
        np.savez(cache, descs=descriptors, poses=poses)
        print(f"[compare] cached descriptors to {cache}")
    return descriptors, poses


def _build_map(
    descriptors: np.ndarray, poses: np.ndarray, alpha: list[float],
    seed: int, window_size: int, valley_k: float,
) -> tuple[TopoGraph, dict[int, np.ndarray]]:
    sim = OdometrySimulator(OdometryNoiseParams(*alpha), seed=seed)
    mapper = TopologicalMapper(window_size=window_size, valley_k=valley_k)
    dummy = np.zeros((8, 8, 3), dtype=np.uint8)
    node_gt: dict[int, np.ndarray] = {}
    for i in range(len(poses)):
        gt = (float(poses[i, 0]), float(poses[i, 1]), float(poses[i, 2]))
        odom, cov = sim.step(gt)
        touched = mapper.process_frame(dummy, descriptors[i], odom, cov, gt_pose=gt)
        if touched is not None:
            # Use the node's representative-frame GT (set by the mapper), so
            # the comparison measures odometric distortion rather than the
            # medoid/valley-frame mismatch. Falls back to this frame's GT.
            node = mapper.graph.nodes[touched]
            gt_xy = (
                np.array(node.gt_pose[:2], dtype=np.float64)
                if node.gt_pose is not None
                else poses[i, :2].astype(np.float64)
            )
            node_gt.setdefault(touched, gt_xy)
    mapper.finalize()
    return mapper.graph, node_gt


def _panel(ax, graph: TopoGraph, node_gt: dict[int, np.ndarray],
           gt_xy: np.ndarray, title: str) -> None:
    ids = sorted(graph.nodes)
    map_pos = graph.positions()
    gt_pos = np.array([node_gt[i] for i in ids], dtype=np.float64)
    fit = _fit_se2(map_pos, gt_pos)
    aligned = map_pos @ fit[0].T + fit[1] if fit else map_pos
    index = {n: k for k, n in enumerate(ids)}
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="0.8", lw=0.8, zorder=1)
    for a, b in graph.edges():
        ka, kb = index[a], index[b]
        ax.plot([aligned[ka, 0], aligned[kb, 0]], [aligned[ka, 1], aligned[kb, 1]],
                color="steelblue", lw=1.0, zorder=3)
    ax.scatter(aligned[:, 0], aligned[:, 1], c="red", s=28,
               edgecolors="black", linewidths=0.4, zorder=5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-trajectory", required=True)
    parser.add_argument("--extractor", default="finetuned:src")
    parser.add_argument("--model-name", default="visual_encoder_dino_contrastive_dim128_best")
    parser.add_argument("--encoder-path", default="/workspace/encoder")
    parser.add_argument("--alpha", nargs=4, type=float, default=[0.025, 0.005, 0.01, 0.0025])
    parser.add_argument("--valley-k", type=float, default=1.5)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--odom-seed", type=int, default=17)
    parser.add_argument("--cache", default="")
    parser.add_argument("--out", default="map_compare.png")
    args = parser.parse_args()

    descriptors, poses = _load_descriptors(
        args.gt_trajectory, args.extractor, args.model_name,
        args.encoder_path, args.cache,
    )
    descriptors = descriptors / np.maximum(
        np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-12
    )
    gt_xy = poses[:, :2].astype(np.float64)

    configs = [
        ("no odometry drift (alpha=0)", [0.0, 0.0, 0.0, 0.0]),
        (f"with odometry drift (alpha={args.alpha})", args.alpha),
    ]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    for ax, (label, alpha) in zip(axes, configs):
        graph, node_gt = _build_map(
            descriptors, poses, alpha, args.odom_seed,
            args.window_size, args.valley_k,
        )
        gm = graph_metrics(graph, gt_xy, node_gt_xy=node_gt)
        _panel(ax, graph, node_gt, gt_xy,
               f"{label}\n{gm.n_nodes} nodes, {gm.n_edges} edges, "
               f"placement RMSE={gm.node_placement_rmse:.2f} m, "
               f"false-merge={gm.false_merge_rate:.2f}")
        print(f"[compare] {label}: nodes={gm.n_nodes} edges={gm.n_edges} "
              f"placement_rmse={gm.node_placement_rmse:.2f}m "
              f"false_merge={gm.false_merge_rate:.3f}")
    fig.suptitle(f"valley_k={args.valley_k}, window={args.window_size}, seed={args.odom_seed}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] saved {args.out}")


if __name__ == "__main__":
    main()
