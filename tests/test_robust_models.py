from __future__ import annotations

import numpy as np
import pytest

from fcr.advanced_models import GeometryLogisticModel
from fcr.robust_models import DampedGeometryLogisticModel
from fcr.schema import ConnectomeSample


def _synthetic_sample(*, rare: bool = False) -> ConnectomeSample:
    rng = np.random.default_rng(2207 if rare else 707)
    n = 6000 if rare else 1200
    source_xyz = rng.normal(size=(n, 3))
    delta = rng.normal(scale=0.7, size=(n, 3))
    if rare:
        delta[:, 1] = 0.998 * delta[:, 0] + 0.002 * rng.normal(size=n)
        delta[:, 2] = 0.995 * delta[:, 0] + 0.005 * rng.normal(size=n)
    target_xyz = source_xyz + delta
    distance = np.linalg.norm(delta, axis=1)
    logits = (
        (-8.0 if rare else -1.7)
        - 0.8 * np.log1p(distance)
        + 0.35 * delta[:, 0]
        - 0.2 * source_xyz[:, 1]
    )
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    connected = (rng.random(n) < probabilities).astype(np.int8)
    if connected.sum() == 0:
        connected[0] = 1
    return ConnectomeSample(
        source=np.arange(n, dtype=np.int64),
        target=np.arange(n, dtype=np.int64) + n,
        source_type=np.zeros(n, dtype=np.int8),
        target_type=np.zeros(n, dtype=np.int8),
        distance=distance,
        connected=connected,
        source_xyz=source_xyz,
        target_xyz=target_xyz,
    )


def test_damped_solver_is_deterministic_and_monotone() -> None:
    sample = _synthetic_sample()
    first = DampedGeometryLogisticModel(feature_set="spatial", l2=10.0).fit(sample)
    second = DampedGeometryLogisticModel(feature_set="spatial", l2=10.0).fit(sample)

    assert first.converged_ is True
    assert second.converged_ is True
    assert np.array_equal(first.coefficients_, second.coefficients_)
    assert np.array_equal(first.predict_proba(sample), second.predict_proba(sample))
    assert first.objective_history_ == second.objective_history_
    assert all(
        current <= previous
        for previous, current in zip(
            first.objective_history_,
            first.objective_history_[1:],
            strict=True,
        )
    )


def test_damped_solver_converges_on_rare_correlated_problem() -> None:
    sample = _synthetic_sample(rare=True)
    model = DampedGeometryLogisticModel(feature_set="spatial", l2=1.0).fit(sample)
    probabilities = model.predict_proba(sample)

    assert model.converged_ is True
    assert model.iterations_ <= 500
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))
    assert all(
        current <= previous
        for previous, current in zip(
            model.objective_history_,
            model.objective_history_[1:],
            strict=True,
        )
    )


def test_damped_solver_matches_original_when_original_converges() -> None:
    sample = _synthetic_sample()
    original = GeometryLogisticModel(feature_set="relative", l2=1.0).fit(sample)
    repaired = DampedGeometryLogisticModel(feature_set="relative", l2=1.0).fit(sample)

    assert original.converged_ is True
    assert repaired.converged_ is True
    assert repaired.model_bits() == original.model_bits()
    assert np.max(np.abs(repaired.predict_proba(sample) - original.predict_proba(sample))) < 1e-8
    assert repaired.coefficients_ == pytest.approx(original.coefficients_, abs=1e-7, rel=1e-7)
