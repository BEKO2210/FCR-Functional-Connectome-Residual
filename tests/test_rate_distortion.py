import numpy as np

from fcr.rate_distortion import (
    adjacency_from_edges,
    normalized_mse,
    structural_rate_distortion_curve,
)


def test_adjacency_direction_is_target_by_source():
    source = np.array([0, 1])
    target = np.array([1, 2])
    connected = np.array([1, 0])
    a = adjacency_from_edges(3, source, target, connected)
    assert a[1, 0] == 1.0
    assert a[2, 1] == 0.0


def test_full_retention_has_zero_distortion():
    source = np.array([0, 1, 2, 0, 2])
    target = np.array([1, 2, 0, 2, 1])
    y = np.ones(5, dtype=int)
    p = np.array([0.9, 0.2, 0.7, 0.1, 0.8])
    curve = structural_rate_distortion_curve(
        n_nodes=3,
        source=source,
        target=target,
        connected=y,
        probabilities=p,
        fractions=(0.5, 1.0),
    )
    assert curve[-1].distortion < 1e-14


def test_normalized_mse_identical_is_zero():
    x = np.arange(9, dtype=float).reshape(3, 3)
    assert normalized_mse(x, x) == 0.0
