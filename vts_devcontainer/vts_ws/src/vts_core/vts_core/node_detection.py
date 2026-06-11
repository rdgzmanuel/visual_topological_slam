"""Node detection via algebraic connectivity with adaptive valley detection.

The sliding-window affinity matrix and normalized-Laplacian machinery is kept
from the thesis (it is the core contribution), but the peak/valley logic is
reworked: the two hand-tuned proportions (``gamma_proportion`` and
``delta_proportion``) are replaced by a single robust-statistics rule. A
valley is confirmed when the eigenvalue series rises above its running
minimum by more than ``k`` median-absolute-deviations (MAD) of the series'
recent variation, with ``k = 3`` — the conventional robust outlier constant.
This adapts automatically to the signal's noise level in each environment
instead of requiring per-dataset tuning, which directly serves the
"threshold-free" goal: the remaining constant has a standard statistical
meaning rather than being a per-environment magic number.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import eigsh

_SPARSE_MATRIX_THRESHOLD: int = 50
_ROBUST_K: float = 3.0
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
        self.max_degree_index: int = 0
        self._max_degree_value: float = float("-inf")

    @property
    def window_length(self) -> int:
        """Current number of images in the window."""
        return 0 if self.window_features is None else int(self.window_features.shape[0])

    def reset_representative(self) -> None:
        """Reset the running best-representative tracker after a node is made."""
        self._max_degree_value = float("-inf")

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
        self.affinity[:-1, -1] = similarities
        self.affinity[-1, :-1] = similarities
        self.affinity[-1, -1] = 0.0

        degrees: np.ndarray = self.affinity.sum(axis=1)
        max_idx: int = int(np.argmax(degrees))
        if degrees[max_idx] > self._max_degree_value:
            self._max_degree_value = float(degrees[max_idx])
            self.max_degree_index = max_idx

        if self.window_length < 2:
            return None

        inv_sqrt: np.ndarray = 1.0 / np.sqrt(np.maximum(degrees, 1e-10))
        laplacian: np.ndarray = np.eye(self.affinity.shape[0], dtype=np.float64) - (
            inv_sqrt[:, None] * self.affinity * inv_sqrt[None, :]
        )

        if laplacian.shape[0] > _SPARSE_MATRIX_THRESHOLD:
            sparse = scipy.sparse.csr_matrix(laplacian)
            values, _ = eigsh(sparse, k=2, which="SM")
            lambda_2: float = float(np.sort(values)[1])
        else:
            values = np.linalg.eigvalsh(laplacian)
            lambda_2 = float(values[1])

        self.eigenvalues.append(lambda_2)
        return lambda_2


class AdaptiveValleyDetector:
    """Streaming valley detector with data-driven prominence.

    State machine identical in spirit to the thesis' peak/valley search, but
    the required prominence is ``max(k * MAD_TO_STD * MAD(diffs), eps)``
    computed over a rolling buffer of first differences of lambda_2 — i.e.,
    the detector asks for a rise that is statistically significant relative
    to the signal's own recent volatility.
    """

    def __init__(self, history: int = 100) -> None:
        """
        Args:
            history: Length of the rolling buffer of lambda_2 differences.
        """
        self._diffs: deque[float] = deque(maxlen=history)
        self._previous: float | None = None
        self._running_min: float = float("inf")
        self._running_min_index: int = -1
        self._descending: bool = False
        self._sample_index: int = -1

    def _prominence(self) -> float:
        if len(self._diffs) < 5:
            return float("inf")  # not enough evidence yet: never trigger
        diffs: np.ndarray = np.abs(np.array(self._diffs, dtype=np.float64))
        mad: float = float(np.median(np.abs(diffs - np.median(diffs))))
        return max(_ROBUST_K * _MAD_TO_STD * mad, 1e-6)

    def step(self, lambda_2: float) -> int | None:
        """Feed one lambda_2 sample.

        Args:
            lambda_2: New algebraic-connectivity value.

        Returns:
            The sample index of a confirmed valley, or None.
        """
        self._sample_index += 1

        if self._previous is not None:
            self._diffs.append(lambda_2 - self._previous)
        self._previous = lambda_2

        if lambda_2 < self._running_min:
            self._running_min = lambda_2
            self._running_min_index = self._sample_index
            self._descending = True
            return None

        if self._descending and lambda_2 > self._running_min + self._prominence():
            valley_index: int = self._running_min_index
            self._descending = False
            self._running_min = lambda_2
            self._running_min_index = self._sample_index
            return valley_index

        return None
