"""Visual place-recognition recall for the PRISM-TopoMap comparison.

Evaluates the *encoder* (not the topological map) with PRISM-TopoMap's
place-recognition protocol (their Table II): rank a database of frames by
descriptor similarity to each query and count a hit@k if any top-k neighbour is
physically within a distance threshold of the query. This isolates the place
recognition quality on COLD so it can be tabled next to PRISM's AR@1 / AR@5.

Because the comparison is cross-dataset and cross-sensor (PRISM report Habitat,
multi-camera + LiDAR; we report COLD, monocular), this is a metric-CLASS
comparison, not a head-to-head — and the fair visual baseline is PRISM's
RGB-only models (GeM/NetVLAD/MixVPR/CosPlace), not the multimodal MSSPlace-G.

Two protocols are reported:

- ``cross-condition`` (the meaningful COLD story): database and queries are the
  SAME physical route under DIFFERENT lighting/time (e.g. db=night, query=
  cloudy/sunny). Tests appearance invariance; no self-matches, so no exclusion.
- ``within`` (mirrors PRISM exactly): query and database are the same run;
  near-in-time frames are excluded so a frame cannot trivially match itself.

Descriptors are cached per traversal (extraction is the only slow step), so
re-running with different thresholds/protocols is instant.

    python3 -m vts_evaluation.place_recognition_eval \
        --db   /workspace/encoder/seq_data/cold-freiburg_part_a_seq2_night1/std_cam \
        --queries /workspace/encoder/seq_data/cold-freiburg_part_a_seq2_cloudy1/std_cam \
                  /workspace/encoder/seq_data/cold-freiburg_part_a_seq2_sunny1/std_cam \
        --extractor finetuned:src \
        --model-name visual_encoder_dino_contrastive_dim128_best \
        --encoder-path /workspace/encoder \
        --cache-dir /tmp/pr_desc --thresholds 2 5 --within

Needs torch + the encoder (run it in the container).
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np

from vts_core.metrics import place_recognition_recall

_POSE = re.compile(r"_x(?P<x>-?\d+\.?\d*)_y(?P<y>-?\d+\.?\d*)_a(?P<a>-?\d+\.?\d*)")


def _traversal_name(images_dir: str) -> str:
    """Stable identifier for a traversal from its std_cam path."""
    parts = [p for p in images_dir.split(os.sep) if p]
    # .../cold-<...>/std_cam  ->  cold-<...>
    return parts[-2] if len(parts) >= 2 else parts[-1]


def load_descriptors(
    images_dir: str,
    extractor_spec: str,
    model_name: str,
    encoder_path: str,
    cache_dir: str,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (descriptors (N,d), poses (N,2)); cache per traversal to disk."""
    name: str = _traversal_name(images_dir)
    cache: str = os.path.join(cache_dir, f"{name}_stride{stride}.npz") if cache_dir else ""
    if cache and os.path.exists(cache):
        data = np.load(cache)
        print(f"[pr] loaded cached descriptors for {name} from {cache}")
        return data["descs"], data["poses"]

    import cv2

    from vts_core.features import build_extractor

    names = [n for n in sorted(os.listdir(images_dir)) if _POSE.search(n)][::stride]
    if not names:
        raise RuntimeError(f"No pose-encoded COLD images in {images_dir}")
    poses = np.array(
        [[float(_POSE.search(n).group(g)) for g in ("x", "y")] for n in names],
        dtype=np.float64,
    )
    extractor = build_extractor(extractor_spec, model_name, encoder_path)
    descs: list[np.ndarray] = []
    for i, fname in enumerate(names):
        image = cv2.imread(os.path.join(images_dir, fname), cv2.IMREAD_COLOR)
        descs.append(extractor.extract(image))
        if i % 250 == 0:
            print(f"[pr] {name}: extracted {i}/{len(names)}", flush=True)
    descriptors = np.asarray(descs, dtype=np.float32)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, descs=descriptors, poses=poses)
        print(f"[pr] cached descriptors for {name} to {cache}")
    return descriptors, poses


def _fmt(recall: dict[str, float], k_values: tuple[int, ...]) -> str:
    cells = [f"R@{k}={100 * recall[f'recall_at_{k}']:.2f}" for k in k_values]
    return "  ".join(cells) + f"  (n={recall['n_queries']})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="database traversal std_cam dir")
    parser.add_argument(
        "--queries", nargs="*", default=[],
        help="cross-condition query traversal std_cam dirs (same route, other "
             "lighting). Omit to run only the within-traversal protocol.",
    )
    parser.add_argument("--extractor", default="finetuned:src")
    parser.add_argument("--model-name", default="visual_encoder_dino_contrastive_dim128_best")
    parser.add_argument("--encoder-path", default="/workspace/encoder")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--stride", type=int, default=1, help="subsample frames")
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[2.0, 5.0],
        help="positive-match distance thresholds in metres (PRISM use 5)",
    )
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5])
    parser.add_argument(
        "--within", action="store_true",
        help="also report within-traversal recall on the database run",
    )
    parser.add_argument(
        "--exclude-window", type=int, default=50,
        help="within-traversal: frames within +/- this index are excluded as "
             "self-matches (scale to your frame rate so it spans a few metres)",
    )
    parser.add_argument("--out", default="", help="optional JSON results path")
    args = parser.parse_args()

    results: dict[str, object] = {
        "database": "", "thresholds": args.thresholds, "k": args.k,
        "within": {}, "cross_condition": {},
    }

    k_values: tuple[int, ...] = tuple(args.k)
    db_desc, db_xy = load_descriptors(
        args.db, args.extractor, args.model_name, args.encoder_path,
        args.cache_dir, args.stride,
    )
    db_name: str = _traversal_name(args.db)
    results["database"] = db_name
    print(f"\n=== Place-recognition recall (database: {db_name}, {len(db_xy)} frames) ===")

    for threshold in args.thresholds:
        key: str = f"{threshold:g}m"
        results["within"][key] = {}
        results["cross_condition"][key] = {}
        print(f"\n-- positive match within {threshold:.1f} m --")
        if args.within:
            idx = np.arange(len(db_xy))
            rec = place_recognition_recall(
                db_desc, db_xy, db_desc, db_xy, k_values, threshold,
                query_index=idx, db_index=idx, exclude_window=args.exclude_window,
            )
            results["within"][key][db_name] = rec
            print(f"within   {db_name:<45} {_fmt(rec, k_values)}")

        aggregated: list[dict[str, float]] = []
        for query_dir in args.queries:
            q_desc, q_xy = load_descriptors(
                query_dir, args.extractor, args.model_name, args.encoder_path,
                args.cache_dir, args.stride,
            )
            rec = place_recognition_recall(
                q_desc, q_xy, db_desc, db_xy, k_values, threshold
            )
            aggregated.append(rec)
            results["cross_condition"][key][_traversal_name(query_dir)] = rec
            print(f"x-cond   {_traversal_name(query_dir):<45} {_fmt(rec, k_values)}")
        if len(aggregated) > 1:
            mean = {
                f"recall_at_{k}": float(
                    np.mean([r[f"recall_at_{k}"] for r in aggregated])
                )
                for k in k_values
            }
            mean["n_queries"] = int(sum(r["n_queries"] for r in aggregated))
            results["cross_condition"][key]["AVERAGE"] = mean
            print(f"x-cond   {'AVERAGE (AR@k)':<45} {_fmt(mean, k_values)}")

    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[pr] results written to {args.out}")


if __name__ == "__main__":
    main()
