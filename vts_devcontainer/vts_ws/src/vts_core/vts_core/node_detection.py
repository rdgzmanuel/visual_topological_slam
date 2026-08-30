"""Node detection from a non-negative visual affinity graph.

Cosine similarity is first clamped to ``[0, 1]``.  This is essential: a
normalized graph Laplacian assumes non-negative edge weights, whereas raw
cosine similarity may be negative.  Isolated (zero-degree) vertices use the
standard convention ``D^{-1/2}_{ii}=0`` and ``L_{ii}=0``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

_MAD_TO_STD: float = 1.4826  # consistency constant for Gaussian data


class ConnectivityMonitor:
    """Maintains the windowed affinity graph and its algebraic connectivity."""

    def __init__(self, window_size: int) -> None:
        """
        Args:
            window_size: Number of most recent images in the similarity graph.
        """
        self._n: int = window_size
        self.window_features: np.ndarray | None = None
        self.affinity: np.ndarray | None = None
        self.eigenvalues: list[float] = []

    @property
    def window_length(self) -> int:
        """Current number of images in the window."""
        return 0 if self.window_features is None else int(self.window_features.shape[0])

    def update(self, descriptor: np.ndarray) -> float | None:
        """Add a new image descriptor and return the new lambda_2 (if defined).

        Args:
            descriptor: L2-normalized feature vector of the new image.

        Returns:
            Second-smallest eigenvalue of the symmetric normalized Laplacian,
            or None while the window has fewer than 2 images.
        """
        descriptor = descriptor.astype(np.float32)

        if self.window_features is None:
            self.window_features = descriptor[None, :]
            self.affinity = np.zeros((1, 1), dtype=np.float32)
            return None

        if self.window_length < self._n:
            self.affinity = np.pad(self.affinity, ((0, 1), (0, 1)), mode="constant")
            self.window_features = np.vstack([self.window_features, descriptor])
        else:
            self.affinity[:-1, :-1] = self.affinity[1:, 1:]
            self.affinity[-1, :] = 0.0
            self.affinity[:, -1] = 0.0
            self.window_features = np.vstack(
                [self.window_features[1:], descriptor]
            )

        similarities: np.ndarray = self.window_features[:-1] @ descriptor
        weights: np.ndarray = np.clip(similarities, 0.0, 1.0)
        self.affinity[:-1, -1] = weights
        self.affinity[-1, :-1] = weights
        self.affinity[-1, -1] = 0.0

        degrees: np.ndarray = self.affinity.sum(axis=1)

        if self.window_length < 2:
            return None

        nonzero: np.ndarray = degrees > 1e-12
        inv_sqrt: np.ndarray = np.zeros_like(degrees, dtype=np.float64)
        inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        laplacian: np.ndarray = np.eye(self.affinity.shape[0], dtype=np.float64) - (
            inv_sqrt[:, None] * self.affinity * inv_sqrt[None, :]
        )
        # For isolated vertices use L_ii=0, so they contribute a zero
        # eigenvalue rather than an artificial unit self-penalty.
        laplacian[~nonzero, ~nonzero] = 0.0

        # The configured window is small (30 by default), so NumPy's dense
        # symmetric solver is simpler and faster than maintaining a separate
        # sparse dependency/path.
        values = np.linalg.eigvalsh(laplacian)
        lambda_2: float = max(float(values[1]), 0.0)

        self.eigenvalues.append(lambda_2)
        return lambda_2


class AdaptiveValleyDetector:
    """Streaming valley detector: thesis hysteresis structure, adaptive delta.

    State machine identical to the original implementation (alternate between
    seeking a local maximum and a local minimum, switching when the signal
    moves by ``delta`` against the current extremum), with two evidence-based
    changes validated against real COLD lambda_2 series:

    1. ``delta`` is adaptive: ``k * 1.4826 * MAD`` of the lambda_2 *values*
       over a rolling window (k = 1.5), replacing the hand-tuned
       ``delta_proportion``. The values' dispersion tracks the oscillation
       amplitude of the signal in the current environment; first differences
       (a previous design) measure only step noise and are uninformative
       about valley depth.
    2. The ``gamma`` minimum-peak-height gate is removed: in its adaptive
       form it vetoed legitimate transitions whose preceding peak sat at the
       plateau level (~ the rolling median), reducing recall from ~0.94 to
       ~0.81 on real data. The hysteresis alone provides noise immunity.

    A previous version of this class reset its running minimum to an absolute
    level after each confirmation, silently degenerating into a global-minimum
    detector (7 firings on a series containing ~30 valleys). The two-extremum
    hysteresis structure does not have that failure mode.
    """

    def __init__(self, k: float = 1.5, history: int = 300, warmup: int = 30) -> None:
        """
        Args:
            k: Robust scale multiplier for the adaptive delta.
            history: Rolling window (samples) over which the values' MAD is
                computed; at 5 Hz the default spans about one minute.
            warmup: Minimum samples before the detector may fire.
        """
        if k <= 0.0:
            raise ValueError("k must be positive")
        if history < 2:
            raise ValueError("history must be at least 2")
        if warmup < 2 or warmup > history:
            raise ValueError("warmup must lie in [2, history]")
        self._k: float = k
        self._values: deque[float] = deque(maxlen=history)
        self._warmup: int = warmup
        self._look_for_max: bool = True
        self._max_value: float = float("-inf")
        self._min_value: float = float("inf")
        self._min_index: int = 0
        self._sample_index: int = -1
        self.last_latency_samples: int | None = None
        self.latencies: list[int] = []

    def _delta(self) -> float:
        if len(self._values) < self._warmup:
            return float("inf")  # not enough evidence yet: never trigger
        values: np.ndarray = np.array(self._values, dtype=np.float64)
        mad: float = float(np.median(np.abs(values - np.median(values))))
        return max(self._k * _MAD_TO_STD * mad, 1e-6)

    @property
    def warmup(self) -> int:
        """Minimum number of lambda_2 samples required before a detection."""
        return self._warmup

    def step(self, lambda_2: float) -> int | None:
        """Feed one lambda_2 sample.

        Args:
            lambda_2: New algebraic-connectivity value.

        Returns:
            The sample index of a confirmed valley, or None.
        """
        self._sample_index += 1
        self._values.append(lambda_2)
        delta: float = self._delta()

        if self._look_for_max:
            if lambda_2 > self._max_value:
                self._max_value = lambda_2
            if lambda_2 < self._max_value - delta:
                self._look_for_max = False
                self._min_value = lambda_2
                self._min_index = self._sample_index
            return None

        if lambda_2 < self._min_value:
            self._min_value = lambda_2
            self._min_index = self._sample_index
        if lambda_2 > self._min_value + delta:
            self._look_for_max = True
            self._max_value = lambda_2
            self.last_latency_samples = self._sample_index - self._min_index
            self.latencies.append(self.last_latency_samples)
            return self._min_index
        return None


class FixedValleyDetector:
    """Two-extremum hysteresis detector with a constant prominence."""

    def __init__(self, delta: float, warmup: int = 30) -> None:
        """
        Args:
            delta: Absolute lambda_2 recovery required to confirm a valley.
            warmup: Minimum samples before the detector may enter a valley.
        """
        if not np.isfinite(delta) or delta <= 0.0:
            raise ValueError("delta must be positive and finite")
        if warmup < 2:
            raise ValueError("warmup must be at least 2")
        self._delta: float = delta
        self._warmup: int = warmup
        self._look_for_max: bool = True
        self._max_value: float = float("-inf")
        self._min_value: float = float("inf")
        self._min_index: int = 0
        self._sample_index: int = -1
        self.last_latency_samples: int | None = None
        self.latencies: list[int] = []

    @property
    def warmup(self) -> int:
        """Minimum number of lambda_2 samples required before detection."""
        return self._warmup

    def step(self, lambda_2: float) -> int | None:
        """Feed one lambda_2 sample and return a confirmed valley index."""
        self._sample_index += 1

        if self._look_for_max:
            if lambda_2 > self._max_value:
                self._max_value = lambda_2
            if (
                self._sample_index + 1 >= self._warmup
                and lambda_2 < self._max_value - self._delta
            ):
                self._look_for_max = False
                self._min_value = lambda_2
                self._min_index = self._sample_index
            return None

        if lambda_2 < self._min_value:
            self._min_value = lambda_2
            self._min_index = self._sample_index
        if lambda_2 > self._min_value + self._delta:
            self._look_for_max = True
            self._max_value = lambda_2
            self.last_latency_samples = self._sample_index - self._min_index
            self.latencies.append(self.last_latency_samples)
            return self._min_index
        return None
