"""Offline driver for PRISM-TopoMap on COLD input bundles.

Feeds a converted COLD sequence (see tools/convert_cold.py) through the
unmodified PRISM-TopoMap algorithm (TopoSLAMModel) exactly the way the
authors' ROS node does, but with a simulated clock so runs are deterministic:

- one ``TopoSLAMModel.update(gt_pose, odom_pose, None, None, cloud, None)``
  call per selected frame (the node's pointcloud callback), and
- one ``localizer.localize()`` call every ``localization_frequency`` seconds
  of *data* time (the node's rospy.Timer).

The only code difference against upstream is the auto device selection
(vendor/device_autodetect.patch): CUDA when available, CPU otherwise.

Runs inside the authors' Docker image (kirillmouraviev/prism-topomap).

Usage:
    python3 run_prism.py --env freiburg_a [--max-frames N] [--workspace /workspace]
"""

import argparse
import json
import os
import resource
import sys
import time
import types

import numpy as np
import yaml


def install_stubs():
    """Stub the two imports the core scripts pull in but never use offline.

    - ros_numpy: utils.py only uses it inside get_xyz_coords_from_msg (the
      ROS message decoder, which this driver replaces).
    - memory_profiler: topo_graph.py imports `profile` but never applies it.
    """
    if "ros_numpy" not in sys.modules:
        sys.modules["ros_numpy"] = types.ModuleType("ros_numpy")
    if "memory_profiler" not in sys.modules:
        mp = types.ModuleType("memory_profiler")
        mp.profile = lambda func: func
        sys.modules["memory_profiler"] = mp


def scan_to_points(ranges, angles, r_min=0.05, r_max=50.0):
    """Convert one 181-beam scan into an Nx3 planar point cloud (z=0)."""
    ok = (ranges > r_min) & (ranges < r_max)
    a = angles[ok]
    r = ranges[ok]
    return np.stack([r * np.cos(a), r * np.sin(a), np.zeros(len(a))], axis=1)


def rel_se2(pose_from, pose_to):
    """SE(2) transform of pose_to expressed in the frame of pose_from."""
    dx = pose_to[0] - pose_from[0]
    dy = pose_to[1] - pose_from[1]
    c, s = np.cos(-pose_from[2]), np.sin(-pose_from[2])
    return (
        c * dx - s * dy,
        s * dx + c * dy,
        pose_to[2] - pose_from[2],
    )


def aggregate_cloud(frames, cur_odom):
    """Motion-compensate buffered scans into the current sensor frame.

    frames: list of (odom_pose, points Nx3) in each scan's own frame.
    Points of older scans are re-expressed in the frame of ``cur_odom``
    using the relative odometry transform.
    """
    out = []
    for odom_pose, pts in frames:
        tx, ty, ttheta = rel_se2(cur_odom, odom_pose)
        c, s = np.cos(ttheta), np.sin(ttheta)
        moved = pts.copy()
        moved[:, 0] = pts[:, 0] * c - pts[:, 1] * s + tx
        moved[:, 1] = pts[:, 0] * s + pts[:, 1] * c + ty
        out.append(moved)
    return np.concatenate(out, axis=0)


def count_edges(graph):
    return sum(len(adj) for adj in graph.adj_lists) // 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True,
                        choices=["freiburg_a", "freiburg_ext",
                                 "saarbruecken_a", "saarbruecken_ext"])
    parser.add_argument("--workspace", default="/workspace",
                        help="Mounted prism_comparison directory")
    parser.add_argument("--config", default=None,
                        help="Path to config yaml (default: <workspace>/config/cold.yaml)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Process only the first N selected frames (for smoke tests)")
    parser.add_argument("--out", default=None,
                        help="Output dir (default: <workspace>/results/<env>)")
    args = parser.parse_args()

    ws = args.workspace
    config_path = args.config or os.path.join(ws, "config", "cold.yaml")
    out_dir = args.out or os.path.join(ws, "results", args.env)
    os.makedirs(out_dir, exist_ok=True)

    install_stubs()
    sys.path.insert(0, os.path.join(ws, "vendor", "prism-topomap", "scripts"))

    with open(config_path) as f:
        config = yaml.safe_load(f)
    driver_cfg = config.get("driver", {})
    agg_window = float(driver_cfg.get("scan_aggregation_window_s", 1.0))
    update_rate = float(driver_cfg.get("update_rate_hz", 5.0))
    # Optional: extrude the planar scan at several z heights so the cloud
    # mimics the vertical wall structure a 3D LiDAR would see. Affects only
    # the place-recognition input; the occupancy-grid footprint is identical.
    z_heights = [float(z) for z in driver_cfg.get("z_replicate_heights", [])]
    loc_freq = float(config["topomap"]["localization_frequency"])

    import torch
    print("Device:", "cuda" if torch.cuda.is_available() else "cpu", flush=True)

    from prism_topomap import TopoSLAMModel
    from utils import apply_pose_shift

    data = np.load(os.path.join(ws, "data", f"{args.env}.npz"), allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    scan_t = data["scan_t"]
    scan_ranges = data["scan_ranges"]
    scan_gt = data["scan_gt"]
    scan_odom = data["scan_odom"]
    scan_label = data["scan_label"]
    angles = np.linspace(meta["angle_min_rad"], meta["angle_max_rad"],
                         meta["beam_count"])

    graph_dir = os.path.join(out_dir, "graph")
    model = TopoSLAMModel(config,
                          path_to_load_graph=None,
                          path_to_save_graph=graph_dir,
                          path_to_save_logs=None)

    frame_log_path = os.path.join(out_dir, "frames.jsonl")
    loc_log_path = os.path.join(out_dir, "localizations.jsonl")
    frame_log = open(frame_log_path, "w")
    loc_log = open(loc_log_path, "w")

    # Frame selection at the requested update rate.
    min_dt = 1.0 / update_rate if update_rate > 0 else 0.0
    buffer = []  # (t, odom_pose, points) within the aggregation window
    last_update_t = -np.inf
    next_loc_t = scan_t[0] + loc_freq
    update_times, loc_times = [], []
    frame_descriptors = []  # one per frames.jsonl line, for PR-recall evaluation
    n_updates = 0
    t_start_wall = time.time()

    for i in range(len(scan_t)):
        t = float(scan_t[i])
        pts = scan_to_points(scan_ranges[i], angles)
        if len(pts) == 0:
            continue
        odom_pose = [float(v) for v in scan_odom[i]]
        gt_pose = [float(v) for v in scan_gt[i]]
        buffer.append((t, odom_pose, pts))
        buffer = [b for b in buffer if t - b[0] <= agg_window]

        # Emulate the localization timer on data time (serialized).
        while t >= next_loc_t:
            t0 = time.perf_counter()
            model.localizer.localize()
            dt_loc = (time.perf_counter() - t0) * 1000.0
            loc_times.append(dt_loc)
            state = model.localizer.get_localized_state()
            matched = state["vertex_ids_matched"]
            loc_log.write(json.dumps({
                "t": t,
                "gt": gt_pose,
                "matched": [int(v) for v in matched] if matched is not None else None,
                "rel_poses": np.asarray(state["rel_poses"]).tolist()
                             if state["rel_poses"] is not None else None,
                "unmatched": [int(v) for v in state["vertex_ids_unmatched"]]
                             if state["vertex_ids_unmatched"] is not None else None,
                "localize_ms": dt_loc,
            }) + "\n")
            next_loc_t += loc_freq

        if t - last_update_t < min_dt:
            continue
        last_update_t = t

        cloud = aggregate_cloud([(b[1], b[2]) for b in buffer], odom_pose)
        if z_heights:
            cloud = np.concatenate(
                [np.c_[cloud[:, 0], cloud[:, 1], np.full(len(cloud), z)]
                 for z in z_heights], axis=0)
        model.current_stamp = t
        n_vertices_before = len(model.graph.vertices)
        n_edges_before = count_edges(model.graph)
        t0 = time.perf_counter()
        model.update(gt_pose, odom_pose, None, None, cloud, None)
        dt_upd = (time.perf_counter() - t0) * 1000.0
        update_times.append(dt_upd)
        n_updates += 1
        frame_descriptors.append(np.asarray(model.cur_desc).reshape(-1).copy())

        est_pose = apply_pose_shift(
            model.last_vertex["pose_for_visualization"], *model.rel_pose_of_vcur
        ) if model.last_vertex is not None else None
        frame_log.write(json.dumps({
            "t": t,
            "gt": gt_pose,
            "odom": odom_pose,
            "label": str(scan_label[i]),
            "est_pose": [float(v) for v in est_pose] if est_pose else None,
            "vcur": int(model.last_vertex_id)
                    if model.last_vertex_id is not None else None,
            "n_vertices": len(model.graph.vertices),
            "n_edges": count_edges(model.graph),
            "new_vertex": len(model.graph.vertices) > n_vertices_before,
            "new_edges": count_edges(model.graph) - n_edges_before,
            "loop_closure": bool(model.found_loop_closure),
            "iou": float(model.cur_iou),
            "cloud_points": int(len(cloud)),
            "update_ms": dt_upd,
        }) + "\n")

        if args.max_frames is not None and n_updates >= args.max_frames:
            print(f"Stopping after {n_updates} frames (--max-frames)", flush=True)
            break

    model.save_graph()
    frame_log.close()
    loc_log.close()
    if frame_descriptors:
        np.save(os.path.join(out_dir, "frame_descriptors.npy"),
                np.stack(frame_descriptors))

    max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    if sys.platform == "linux":
        max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3
    summary = {
        "environment": args.env,
        "sequence": meta["sequence"],
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "n_scans": int(len(scan_t)),
        "n_updates": n_updates,
        "n_localize_calls": len(loc_times),
        "n_vertices": len(model.graph.vertices),
        "n_edges": count_edges(model.graph),
        "update_ms_mean": float(np.mean(update_times)) if update_times else None,
        "update_ms_median": float(np.median(update_times)) if update_times else None,
        "localize_ms_mean": float(np.mean(loc_times)) if loc_times else None,
        "localize_ms_median": float(np.median(loc_times)) if loc_times else None,
        "wall_time_s": time.time() - t_start_wall,
        "max_rss_mb": max_rss_mb,
        "config": {
            "scan_aggregation_window_s": agg_window,
            "update_rate_hz": update_rate,
            "localization_frequency_s": loc_freq,
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
