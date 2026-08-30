"""Plot the algebraic-connectivity (lambda_2) series with detected valleys.

Replays the configured valley detector over a lambda_2 series saved by the
graph builder (``output/<env>/graph_<i>_lambda2.npy``), so the plotted valleys
are exactly what the mapper detected. Used to generate the node-extraction
figure of the paper.

    python3 -m vts_evaluation.plot_lambda2 \
        --lambda2 output/revised/freiburg_a/graph_0_lambda2.npy \
        --valley-k 1.5 --out lambda2_valleys.pdf
"""

from __future__ import annotations

import argparse

import numpy as np

from vts_core.node_detection import AdaptiveValleyDetector, FixedValleyDetector


def detect_valleys(
    series: np.ndarray,
    valley_k: float,
    valley_mode: str = "adaptive",
    valley_delta: float = 0.1,
) -> list[int]:
    """Replay the selected detector over a saved lambda_2 series."""
    if valley_mode == "adaptive":
        detector = AdaptiveValleyDetector(k=valley_k)
    elif valley_mode == "fixed":
        detector = FixedValleyDetector(delta=valley_delta)
    else:
        raise ValueError("valley_mode must be 'adaptive' or 'fixed'")
    valleys: list[int] = []
    for value in series:
        index = detector.step(float(value))
        if index is not None:
            valleys.append(index)
    return valleys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda2", required=True, help="graph_<i>_lambda2.npy path")
    parser.add_argument("--valley-k", type=float, default=1.5,
                        help="detector sensitivity (use the run's config value)")
    parser.add_argument(
        "--valley-mode", choices=("adaptive", "fixed"), default="adaptive"
    )
    parser.add_argument("--valley-delta", type=float, default=0.1)
    parser.add_argument("--out", default="lambda2_valleys.pdf")
    parser.add_argument("--figsize", nargs=2, type=float, default=[8.0, 3.2])
    parser.add_argument(
        "--skip", type=int, default=0,
        help="omit the first N samples from the PLOT (window warm-up "
             "transient); the detector still replays the full series",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: np.ndarray = np.load(args.lambda2)
    valleys: list[int] = detect_valleys(
        series, args.valley_k, args.valley_mode, args.valley_delta
    )
    parameter = (
        f"valley_k={args.valley_k}"
        if args.valley_mode == "adaptive"
        else f"valley_delta={args.valley_delta}"
    )
    print(
        f"[lambda2] {len(series)} samples, {len(valleys)} valleys "
        f"({args.valley_mode}, {parameter})"
    )

    start: int = max(0, args.skip)
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    ax.plot(np.arange(start, len(series)), series[start:], color="tab:blue",
            lw=1.0, label=r"$\lambda_2$")
    shown = [v for v in valleys if v >= start]
    for i, v in enumerate(shown):
        ax.axvline(v, color="0.35", lw=0.9, linestyle="--",
                   label="detected valley" if i == 0 else None)
    ax.set_xlabel("Frame index", fontsize=13)
    ax.set_ylabel(r"Algebraic connectivity $\lambda_2$", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.set_xlim(start, len(series) - 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=14, framealpha=1.0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[lambda2] figure saved to {args.out}")


if __name__ == "__main__":
    main()
