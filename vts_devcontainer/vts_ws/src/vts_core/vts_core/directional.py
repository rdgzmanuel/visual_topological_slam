"""Directional statistics for hyperspherical visual descriptors.

DINO descriptors are L2-normalized, so they live on a unit hypersphere rather
than in unconstrained Euclidean space. This module models the observations of
one topological place with a von Mises--Fisher (vMF) distribution. It has no
learned parameters and depends only on NumPy and the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log, log1p, pi, sqrt

import numpy as np

_MIN_NORM: float = 1e-12
_MAX_CONCENTRATION: float = 1e6


@dataclass(frozen=True)
class VonMisesFisher:
    """Maximum-likelihood directional summary of one visual segment."""

    mean_direction: np.ndarray
    concentration: float
    mean_resultant_length: float
    sample_count: int


def fit_von_mises_fisher(descriptors: np.ndarray) -> VonMisesFisher:
    """Fit a vMF distribution to an ``(n, d)`` descriptor matrix.

    The standard high-dimensional approximation

    ``kappa = r * (d - r**2) / (1 - r**2)``

    is used for the concentration MLE. Concentration cannot be inferred from
    one observation, so singleton segments are represented as uninformative
    (``kappa = 0``). The numerical concentration cap affects only descriptors
    that are indistinguishable at floating-point precision.
    """
    samples = np.asarray(descriptors, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] < 2:
        raise ValueError("descriptors must have shape (n >= 1, d >= 2)")

    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    if np.any(norms <= _MIN_NORM):
        raise ValueError("descriptors must be non-zero")
    samples = samples / norms

    resultant = samples.sum(axis=0)
    resultant_norm = float(np.linalg.norm(resultant))
    sample_count, dimension = samples.shape
    if resultant_norm <= _MIN_NORM:
        mean_direction = samples[0].copy()
        mean_resultant_length = 0.0
    else:
        mean_direction = resultant / resultant_norm
        mean_resultant_length = float(
            np.clip(resultant_norm / sample_count, 0.0, 1.0)
        )

    concentration = 0.0
    if sample_count > 1 and mean_resultant_length > _MIN_NORM:
        r = min(mean_resultant_length, 1.0 - 1e-12)
        concentration = r * (dimension - r * r) / (1.0 - r * r)
        concentration = min(float(concentration), _MAX_CONCENTRATION)

    return VonMisesFisher(
        mean_direction=mean_direction.astype(np.float32),
        concentration=concentration,
        mean_resultant_length=mean_resultant_length,
        sample_count=sample_count,
    )


def _log_vmf_normalizer(dimension: int, concentration: float) -> float:
    """Log normalizer of a vMF density without a SciPy dependency.

    DINOv2-S has dimension 384. For this high-order Bessel function, the
    uniform asymptotic expansion is both stable and accurate. Two correction
    terms are included; the omitted error is of order ``nu**-3`` where
    ``nu = dimension / 2 - 1``. The exact uniform-density limit is used at
    zero concentration.
    """
    if dimension < 3:
        raise ValueError("vMF evidence requires dimension >= 3")
    kappa = max(float(concentration), 0.0)
    log_uniform = (
        lgamma(0.5 * dimension)
        - log(2.0)
        - 0.5 * dimension * log(pi)
    )
    if kappa == 0.0:
        return log_uniform
    if kappa < 1e-3:
        return log_uniform - kappa * kappa / (2.0 * dimension)

    order = 0.5 * dimension - 1.0
    z = kappa / order
    root = sqrt(1.0 + z * z)
    eta = root + log(z) - log1p(root)
    t = 1.0 / root
    u1 = (3.0 * t - 5.0 * t**3) / 24.0
    u2 = (81.0 * t**2 - 462.0 * t**4 + 385.0 * t**6) / 1152.0
    correction = 1.0 + u1 / order + u2 / (order * order)
    log_bessel = (
        -0.5 * log(2.0 * pi * order)
        - 0.25 * log1p(z * z)
        + order * eta
        + log(max(correction, _MIN_NORM))
    )
    return (
        order * log(kappa)
        - 0.5 * dimension * log(2.0 * pi)
        - log_bessel
    )


def vmf_log_overlap_ratio(
    first_direction: np.ndarray,
    first_concentration: float,
    second_direction: np.ndarray,
    second_concentration: float,
) -> float:
    """Return vMF distribution overlap relative to a uniform descriptor.

    A score of zero means that at least one place is visually uninformative.
    Positive values indicate more overlap than two uniform directions;
    strongly incompatible concentrated places yield negative values. The
    score is symmetric and contains no calibrated or hand-picked threshold.
    """
    first = np.asarray(first_direction, dtype=np.float64).reshape(-1)
    second = np.asarray(second_direction, dtype=np.float64).reshape(-1)
    if first.shape != second.shape or first.size < 3:
        raise ValueError("directions must have the same dimension >= 3")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= _MIN_NORM or second_norm <= _MIN_NORM:
        raise ValueError("directions must be non-zero")
    first /= first_norm
    second /= second_norm

    kappa_a = max(float(first_concentration), 0.0)
    kappa_b = max(float(second_concentration), 0.0)
    cosine = float(np.clip(first @ second, -1.0, 1.0))
    combined_squared = (
        kappa_a * kappa_a
        + kappa_b * kappa_b
        + 2.0 * kappa_a * kappa_b * cosine
    )
    combined = sqrt(max(combined_squared, 0.0))
    dimension = first.size
    log_uniform = _log_vmf_normalizer(dimension, 0.0)
    return float(
        _log_vmf_normalizer(dimension, kappa_a)
        + _log_vmf_normalizer(dimension, kappa_b)
        - _log_vmf_normalizer(dimension, combined)
        - log_uniform
    )
