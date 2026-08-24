"""Deterministic synthetic connectomes for tests and null experiments."""

from __future__ import annotations

import numpy as np

from .schema import ConnectomeSample


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_connectome(
    *,
    n_nodes: int = 500,
    n_pairs: int = 40_000,
    n_types: int = 4,
    seed: int = 7,
) -> ConnectomeSample:
    """Generate a sparse graph where type and geometry genuinely predict edges.

    The generator is a null/sanity environment only. It must never be used as
    evidence for biological compression claims.
    """
    if n_nodes < 2 or n_pairs < 1 or n_types < 1:
        raise ValueError("invalid synthetic dimensions")
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, 1.0, size=(n_nodes, 3))
    node_types = rng.integers(0, n_types, size=n_nodes)
    source = rng.integers(0, n_nodes, size=n_pairs)
    target = rng.integers(0, n_nodes, size=n_pairs)
    same = source == target
    while np.any(same):
        target[same] = rng.integers(0, n_nodes, size=int(same.sum()))
        same = source == target

    distance = np.linalg.norm(positions[source] - positions[target], axis=1)
    source_type = node_types[source]
    target_type = node_types[target]

    type_affinity = np.where(source_type == target_type, 1.2, -0.35)
    directional = 0.25 * (source_type - target_type) / max(1, n_types - 1)
    logits = -2.4 - 5.0 * distance + type_affinity + directional
    probability = _sigmoid(logits)
    connected = rng.binomial(1, probability).astype(np.int8)

    return ConnectomeSample(
        source=source,
        target=target,
        source_type=source_type.astype(str),
        target_type=target_type.astype(str),
        distance=distance,
        connected=connected,
    )


def train_test_split(
    sample: ConnectomeSample, *, test_fraction: float = 0.25, seed: int = 11
) -> tuple[ConnectomeSample, ConnectomeSample]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sample))
    cut = int(round(len(sample) * (1.0 - test_fraction)))
    train_mask = np.zeros(len(sample), dtype=bool)
    train_mask[order[:cut]] = True
    return sample.subset(train_mask), sample.subset(~train_mask)
