"""Information-theoretic metrics used by FCR."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _validate_binary_targets(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int8)
    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("y must contain only 0 and 1")
    return y


def _validate_probabilities(p: np.ndarray, n: int) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or len(p) != n:
        raise ValueError("p must be one-dimensional and match y")
    if not np.all(np.isfinite(p)):
        raise ValueError("p must contain only finite values")
    return np.clip(p, _EPS, 1.0 - _EPS)


def binary_code_length_bits(y: np.ndarray, p: np.ndarray) -> float:
    """Ideal arithmetic-code length for Bernoulli observations in bits.

    This is the held-out negative log2 likelihood. It is not, by itself, a
    complete MDL score because model-description cost is tracked separately.
    """
    yv = _validate_binary_targets(y)
    pv = _validate_probabilities(p, len(yv))
    return float(-np.sum(yv * np.log2(pv) + (1 - yv) * np.log2(1.0 - pv)))


def bits_per_sample(total_bits: float, n_samples: int) -> float:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    return float(total_bits) / n_samples


def bits_per_positive(total_bits: float, y: np.ndarray) -> float:
    yv = _validate_binary_targets(y)
    positives = int(yv.sum())
    if positives == 0:
        raise ValueError("bits_per_positive is undefined with zero positives")
    return float(total_bits) / positives


def bernoulli_entropy_bits(p: float) -> float:
    """Binary entropy H(p) in bits."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be between 0 and 1")
    if p in (0.0, 1.0):
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))
