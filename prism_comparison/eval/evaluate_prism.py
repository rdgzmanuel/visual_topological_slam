"""Evaluate PRISM-TopoMap runs on COLD with the same metrics as the VTS paper.

Reads results/<env>/ (graph/graph.json, frames.jsonl, localizations.jsonl,
summary.json, frame_descriptors.npy) plus data/<env>.npz, and computes:

- structural metrics mirroring vts_core.metrics.graph_metrics: nodes, edges,
  connected components, max/mean degree, circuit rank (loop closures),
  coverage (adaptive + @1m/@2m), false-merge rate (edges whose ground-truth
  endpoint distance exceeds the edge's own claimed length by > 3x tolerance,
  tolerance = median nearest-neighbour node spacing);
- online localization RMSE (estimated pose vs ground truth per frame);
- place-recognition AR@1/AR@5 of the minkloc3d descriptors with the exact
  within-run protocol used in the paper (thresholds 2 & 5 m,
  self-match exclusion window of 50 frames at 5 Hz = 10 s), Euclidean ranking
  as in PRISM's faiss IndexFlatL2;
- map size (descriptors-only and on-disk) and timing/RAM from summary.json.

Outputs results/<env>/metrics.json, a per-map trajectory figure, and (with
--all) results/comparison.json + results/comparison_table.tex.

Run on the host: python3 eval/evaluate_prism.py --all
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
ENVS = ["freiburg_a", "freiburg_ext", "saarbruecken_a", "saarbruecken_ext"]


def median_spacing(points: np.ndarray) -> float:
    """Median nearest-neighbour distance (same as vts_core.matching)."""
    n = points.shape[0]
    if n < 2:
        return 1.0
    deltas = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.median(distances.min(axis=1)))


def connected_components(n_vertices: int, edges: list[tuple[int, int]]) -> int:
    parent = list(range(n_vertices))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in edges:
        ra, rb = find(u), find(v)
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n_vertices)})


def recall_at_k(descriptors, xy, k_values=(1, 5), threshold=5.0,
                exclude_window=50):
    """Within-run AR@k, Euclidean ranking (PRISM faiss IndexFlatL2)."""
    q = descriptors.shape[0]
    if q < 2:
        return {f"recall_at_{k}": float("nan") for k in k_values}
    d2 = ((descriptors[:, None, :] - descriptors[None, :, :]) ** 2).sum(-1)
    geo = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    idx = np.arange(q)
    hits = {k: 0 for k in k_values}
    max_k = max(k_values)
    n_queries = 0
    for i in range(q):
        scores = d2[i].copy()
        scores[np.abs(idx - i) <= exclude_window] = np.inf
        if not np.isfinite(scores).any():
            continue
        n_queries += 1
        order = np.argsort(scores)[:max_k]
        correct = geo[i, order] <= threshold
        for k in k_values:
            if bool(correct[:k].any()):
                hits[k] += 1
    out = {f"recall_at_{k}": hits[k] / n_queries for k in k_values}
    out["n_queries"] = n_queries
    return out


def apply_pose_shift(pose, rel_x, rel_y, rel_theta):
    """Same convention as PRISM's utils.apply_pose_shift."""
    x, y, theta = pose
    new_x = x + rel_x * np.cos(-theta) + rel_y * np.sin(-theta)
    new_y = y - rel_x * np.sin(-theta) + rel_y * np.cos(-theta)
    return [new_x, new_y, theta + rel_theta]


def fit_se2_rmse(source: np.ndarray, target: np.ndarray) -> float:
    """RMSE after best-fit SE(2) alignment (mirrors vts_core placement RMSE)."""
    if len(source) < 3:
        return float("nan")
    mu_s, mu_t = source.mean(0), target.mean(0)
    s, t = source - mu_s, target - mu_t
    h = s.T @ t
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, d]) @ u.T
    resid = (s @ rot.T + (mu_t - mu_t)) - t
    return float(np.sqrt(np.mean(np.linalg.norm(resid, axis=1) ** 2)))


def graph_layout_rmse(n_vertices, adj, positions_gt) -> float:
    """Map-frame layout distortion: integrate edge rel poses over a BFS
    spanning tree from vertex 0, then best-fit SE(2) align to the GT vertex
    positions — the direct analog of the VTS node-placement RMSE."""
    if n_vertices < 3:
        return float("nan")
    layout = {0: [positions_gt[0][0], positions_gt[0][1],
                  0.0]}
    from collections import deque
    queue = deque([0])
    visited = {0}
    while queue:
        u = queue.popleft()
        for v, pose in adj[u]:
            if v in visited:
                continue
            visited.add(v)
            layout[v] = apply_pose_shift(layout[u], *pose)
            queue.append(v)
    ids = sorted(visited)
    src = np.array([layout[i][:2] for i in ids])
    tgt = np.array([positions_gt[i] for i in ids])
    return fit_se2_rmse(src, tgt)


def dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / 1e6


def evaluate_env(env: str, make_figure: bool = True) -> dict:
    res = PKG / "results" / env
    graph = json.loads((res / "graph" / "graph.json").read_text())
    frames = [json.loads(l) for l in (res / "frames.jsonl").read_text().splitlines()]
    summary = json.loads((res / "summary.json").read_text())
    data = np.load(PKG / "data" / f"{env}.npz", allow_pickle=True)

    vertices = graph["vertices"]
    adj = graph["edges"]
    n_vertices = len(vertices)
    positions = np.array([v["pose_for_visualization"][:2] for v in vertices])
    edge_set = set()
    edge_claimed_len = {}
    for u, lst in enumerate(adj):
        for v, pose in lst:
            key = (min(u, v), max(u, v))
            edge_set.add(key)
            edge_claimed_len[key] = float(np.hypot(pose[0], pose[1]))
    edges = sorted(edge_set)
    n_edges = len(edges)

    tolerance = median_spacing(positions)
    gt_traj = data["scan_gt"][:, :2]

    # Coverage
    dmat = np.linalg.norm(gt_traj[:, None, :] - positions[None, :, :], axis=2)
    nearest = dmat.min(axis=1)
    coverage = float(np.mean(nearest <= tolerance))
    coverage_1m = float(np.mean(nearest <= 1.0))
    coverage_2m = float(np.mean(nearest <= 2.0))

    # False merges: GT endpoint distance far above the edge's claimed length.
    bad_edges = 0
    for (u, v) in edges:
        gt_dist = float(np.linalg.norm(positions[u] - positions[v]))
        if gt_dist > edge_claimed_len[(u, v)] + 3.0 * tolerance:
            bad_edges += 1
    false_merge_rate = bad_edges / n_edges if n_edges else 0.0

    degrees = [len(adj[i]) for i in range(n_vertices)]
    ncomp = connected_components(n_vertices, edges)
    circuit_rank = max(n_edges - n_vertices + ncomp, 0)

    layout_rmse = graph_layout_rmse(n_vertices, adj, positions)

    # Online localization error (estimated pose vs GT, per processed frame)
    est = np.array([f["est_pose"][:2] for f in frames if f["est_pose"]])
    gt_f = np.array([f["gt"][:2] for f in frames if f["est_pose"]])
    est_err = np.linalg.norm(est - gt_f, axis=1)
    online_rmse = float(np.sqrt(np.mean(est_err ** 2)))

    # Place recognition on per-frame minkloc3d descriptors
    pr = {}
    desc_path = res / "frame_descriptors.npy"
    if desc_path.exists():
        descs = np.load(desc_path)
        frame_xy = np.array([f["gt"][:2] for f in frames])[: len(descs)]
        for thr in (2.0, 5.0):
            pr[f"within_run_at_{int(thr)}m"] = recall_at_k(
                descs, frame_xy, threshold=thr, exclude_window=50
            )

    # Map size
    desc_dim = len(vertices[0]["descriptor"]) if n_vertices else 0
    map_desc_only_mb = n_vertices * desc_dim * 4 / 1e6
    map_disk_mb = dir_size_mb(res / "graph")

    metrics = {
        "environment": env,
        "sequence": summary.get("sequence"),
        "device": summary.get("device"),
        "n_vertices": n_vertices,
        "n_edges": n_edges,
        "n_components": ncomp,
        "circuit_rank_loop_closures": circuit_rank,
        "max_degree": max(degrees, default=0),
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "spatial_tolerance_m": tolerance,
        "coverage": coverage,
        "coverage_1m": coverage_1m,
        "coverage_2m": coverage_2m,
        "false_merge_rate": false_merge_rate,
        "graph_layout_rmse_m": layout_rmse,
        "online_localization_rmse_m": online_rmse,
        "online_localization_p90_m": float(np.percentile(est_err, 90)),
        "place_recognition": pr,
        "map_descriptors_only_mb": map_desc_only_mb,
        "map_disk_mb": map_disk_mb,
        "descriptor_dim": desc_dim,
        "update_ms_mean": summary.get("update_ms_mean"),
        "localize_ms_mean": summary.get("localize_ms_mean"),
        "max_rss_mb": summary.get("max_rss_mb"),
        "n_frames_processed": summary.get("n_updates"),
    }
    (res / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if make_figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 7))
            ax.plot(gt_traj[:, 0], gt_traj[:, 1], color="tab:green", lw=0.8,
                    alpha=0.6, label="ground-truth path")
            if len(est):
                ax.plot(est[:, 0], est[:, 1], color="steelblue", lw=0.7,
                        alpha=0.6, label="PRISM estimate")
            for (u, v) in edges:
                ax.plot(positions[[u, v], 0], positions[[u, v], 1],
                        color="gray", lw=1.0, alpha=0.8, zorder=3)
            ax.scatter(positions[:, 0], positions[:, 1], s=45, c="crimson",
                       zorder=4, label="PRISM locations")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            ax.set_title(f"PRISM-TopoMap on {env}: {n_vertices} locations, "
                         f"{n_edges} edges, Ncomp={ncomp}")
            fig.tight_layout()
            fig.savefig(res / "map.png", dpi=150)
            plt.close(fig)
        except Exception as exc:  # figure is best-effort on minimal hosts
            print(f"[warn] figure for {env} failed: {exc}")

    return metrics


def emit_table(all_metrics: dict[str, dict]) -> str:
    """LaTeX rows: env, nodes, edges, Ncomp, false merge, online RMSE, maxdeg."""
    names = {"freiburg_a": "Freiburg A", "freiburg_ext": "Freiburg Ext.",
             "saarbruecken_a": "Saarbr\\\"ucken A",
             "saarbruecken_ext": "Saarbr\\\"ucken Ext."}
    lines = []
    for env in ENVS:
        m = all_metrics.get(env)
        if not m:
            continue
        lines.append(
            f"{names[env]} & {m['n_vertices']} & {m['n_edges']} & "
            f"{m['n_components']} & {m['false_merge_rate']:.2f} & "
            f"{m['graph_layout_rmse_m']:.2f} & {m['max_degree']} \\\\"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=ENVS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    envs = ENVS if args.all or not args.env else [args.env]
    all_metrics = {}
    for env in envs:
        res = PKG / "results" / env
        if not (res / "summary.json").exists():
            print(f"[skip] {env}: no results yet")
            continue
        m = evaluate_env(env, make_figure=not args.no_figure)
        all_metrics[env] = m
        print(f"{env}: V={m['n_vertices']} E={m['n_edges']} "
              f"Ncomp={m['n_components']} fmerge={m['false_merge_rate']:.2f} "
              f"onlineRMSE={m['online_localization_rmse_m']:.2f} m "
              f"maxdeg={m['max_degree']}")
        if m["place_recognition"]:
            pr5 = m["place_recognition"]["within_run_at_5m"]
            print(f"   PR within-run @5m: AR@1={pr5['recall_at_1']:.3f} "
                  f"AR@5={pr5['recall_at_5']:.3f}")

    if len(all_metrics) > 1:
        (PKG / "results" / "comparison.json").write_text(
            json.dumps(all_metrics, indent=2))
        (PKG / "results" / "comparison_table.tex").write_text(
            emit_table(all_metrics) + "\n")
        print("Wrote results/comparison.json and results/comparison_table.tex")


if __name__ == "__main__":
    main()
