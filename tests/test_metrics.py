import numpy as np
import pytest

from fcr.metrics import bernoulli_entropy_bits, binary_code_length_bits, bits_per_positive


def test_perfect_probabilities_have_near_zero_code_length():
    y = np.array([0, 1, 1, 0])
    p = np.array([1e-12, 1 - 1e-12, 1 - 1e-12, 1e-12])
    assert binary_code_length_bits(y, p) < 1e-8


def test_fair_coin_costs_one_bit_per_sample():
    y = np.array([0, 1, 1, 0, 1])
    p = np.full(len(y), 0.5)
    assert binary_code_length_bits(y, p) == pytest.approx(len(y))


def test_binary_entropy_bounds():
    assert bernoulli_entropy_bits(0.0) == 0.0
    assert bernoulli_entropy_bits(1.0) == 0.0
    assert bernoulli_entropy_bits(0.5) == pytest.approx(1.0)


def test_bits_per_positive_rejects_no_positive_edges():
    with pytest.raises(ValueError):
        bits_per_positive(10.0, np.zeros(5, dtype=int))
