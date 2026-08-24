import numpy as np

from fcr.synthetic import generate_synthetic_connectome, train_test_split


def test_synthetic_is_reproducible():
    a = generate_synthetic_connectome(n_nodes=50, n_pairs=500, seed=5)
    b = generate_synthetic_connectome(n_nodes=50, n_pairs=500, seed=5)
    assert np.array_equal(a.connected, b.connected)
    assert np.allclose(a.distance, b.distance)


def test_split_is_disjoint_and_complete_by_rows():
    sample = generate_synthetic_connectome(n_nodes=50, n_pairs=500, seed=8)
    train, test = train_test_split(sample, test_fraction=0.2, seed=9)
    assert len(train) + len(test) == len(sample)
    assert len(train) == 400
    assert len(test) == 100
