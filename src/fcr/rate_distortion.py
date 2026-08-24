"""Prototype structural-to-functional rate-distortion utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistortionPoint:
    retained_fraction: float
    retained_edges: int
    distortion: float


def adjacency_from_edges(
    n_nodes: int, source: np.ndarray, target: np.ndarray, connected: np.ndarray
) -> np.ndarray:
    matrix = np.zeros((n_nodes, n_nodes), dtype=float)
    mask = np.asarray(connected, dtype=bool)
    matrix[np.asarray(target)[mask], np.asarray(source)[mask]] = 1.0
    return matrix


def stable_weight_matrix(
    adjacency: np.ndarray, *, seed: int = 17, target_radius: float = 0.85
) -> np.ndarray:
    """Assign deterministic random weights and scale to a stable spectral radius."""
    a = np.asarray(adjacency, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency must be square")
    rng = np.random.default_rng(seed)
    w = a * rng.normal(0.0, 1.0, size=a.shape)
    if not np.any(w):
        return w
    radius = float(np.max(np.abs(np.linalg.eigvals(w))))
    if radius > 0:
        w *= target_radius / radius
    return w


def simulate_responses(weights: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    """Run a simple recurrent tanh system under a fixed input sequence."""
    w = np.asarray(weights, dtype=float)
    u = np.asarray(inputs, dtype=float)
    if u.ndim != 2 or w.shape[0] != w.shape[1] or u.shape[1] != w.shape[0]:
        raise ValueError("input and weight dimensions are inconsistent")
    state = np.zeros(w.shape[0], dtype=float)
    out = np.empty_like(u)
    for t in range(len(u)):
        state = np.tanh(w @ state + u[t])
        out[t] = state
    return out


def normalized_mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.shape != cand.shape:
        raise ValueError("reference and candidate shapes must match")
    denom = float(np.mean(ref**2))
    if denom == 0.0:
        return float(np.mean((ref - cand) ** 2))
    return float(np.mean((ref - cand) ** 2) / denom)


def retention_order(
    probabilities: np.ndarray,
    connected: np.ndarray,
    *,
    mode: str,
    seed: int = 23,
) -> np.ndarray:
    """Return positive-edge indices ordered by a non-functional structural rule."""
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0 - 1e-12)
    y = np.asarray(connected, dtype=np.int8)
    positives = np.flatnonzero(y == 1)
    if mode == "residual-first":
        score = -np.log2(p[positives])
        return positives[np.argsort(-score, kind="stable")]
    if mode == "typical-first":
        score = -np.log2(p[positives])
        return positives[np.argsort(score, kind="stable")]
    if mode == "random":
        rng = np.random.default_rng(seed)
        return rng.permutation(positives)
    raise ValueError("mode must be residual-first, typical-first, or random")


def structural_rate_distortion_curve(
    *,
    n_nodes: int,
    source: np.ndarray,
    target: np.ndarray,
    connected: np.ndarray,
    probabilities: np.ndarray,
    fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 1.0),
    mode: str = "residual-first",
    seed: int = 29,
) -> list[DistortionPoint]:
    """Surrogate experiment: retain positive edges and compare recurrent dynamics.

    This is intentionally a test bed, not evidence of biological functional
    equivalence. Real FCR claims require matched functional measurements.
    """
    y = np.asarray(connected, dtype=np.int8)
    order = retention_order(probabilities, y, mode=mode, seed=seed)
    full_a = adjacency_from_edges(n_nodes, source, target, y)
    full_w = stable_weight_matrix(full_a, seed=seed)
    rng = np.random.default_rng(seed + 1)
    inputs = rng.normal(0.0, 0.15, size=(40, n_nodes))
    reference = simulate_responses(full_w, inputs)
    points: list[DistortionPoint] = []
    for fraction in fractions:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("retention fractions must be in [0, 1]")
        k = int(round(len(order) * fraction))
        retained = order[:k]
        partial_y = np.zeros_like(y)
        partial_y[retained] = 1
        partial_a = adjacency_from_edges(n_nodes, source, target, partial_y)
        partial_w = full_w * (partial_a != 0)
        candidate = simulate_responses(partial_w, inputs)
        points.append(DistortionPoint(fraction, k, normalized_mse(reference, candidate)))
    return points
