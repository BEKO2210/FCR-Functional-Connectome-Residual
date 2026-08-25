from __future__ import annotations

import numpy as np
import pytest

import fcr.experiment_008 as experiment_008
from fcr.schema import ConnectomeSample


def _sample() -> ConnectomeSample:
    return ConnectomeSample(
        source=np.array([1, 2], dtype=np.int64),
        target=np.array([2, 1], dtype=np.int64),
        source_type=np.zeros(2, dtype=np.int8),
        target_type=np.zeros(2, dtype=np.int8),
        distance=np.ones(2, dtype=float),
        connected=np.array([1, 0], dtype=np.int8),
        source_xyz=np.zeros((2, 3), dtype=float),
        target_xyz=np.ones((2, 3), dtype=float),
    )


def _evaluation(*, protocol: str = "experiment-005-nested-spatial-cv") -> dict[str, object]:
    return {
        "protocol": protocol,
        "aggregate": {
            "global": {
                "residual_bits": 100.0,
                "two_part_bits": 164.0,
                "residual_reduction_vs_global": 0.0,
                "wins_vs_global": 0,
            },
            "distance": {
                "residual_bits": 80.0,
                "two_part_bits": 400.0,
                "residual_reduction_vs_global": 0.20,
                "wins_vs_global": 5,
            },
            "relative": {
                "residual_bits": 70.0,
                "two_part_bits": 1500.0,
                "residual_reduction_vs_global": 0.30,
                "wins_vs_global": 5,
            },
            "spatial": {
                "residual_bits": 94.0,
                "two_part_bits": 3300.0,
                "residual_reduction_vs_global": 0.06,
                "wins_vs_global": 4,
            },
        },
    }


def test_controlled_rerun_routes_through_damped_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = _sample()
    node_ids = np.array([1, 2], dtype=np.int64)
    coordinates = np.zeros((2, 3), dtype=float)
    called: dict[str, bool] = {}

    def fake_precheck(*args: object) -> list[dict[str, int]]:
        called["precheck"] = True
        return [{"outer_slab": 0, "positive_pairs": 1}]

    def fake_damped(*args: object) -> dict[str, object]:
        called["damped"] = True
        return _evaluation()

    monkeypatch.setattr(experiment_008, "outer_slab_positive_precheck", fake_precheck)
    monkeypatch.setattr(experiment_008, "run_nested_spatial_cv_damped", fake_damped)

    precheck, evaluation, confirmation = experiment_008.run_controlled_h01_rerun(
        sample,
        node_ids,
        coordinates,
    )

    assert called == {"precheck": True, "damped": True}
    assert precheck == [{"outer_slab": 0, "positive_pairs": 1}]
    assert evaluation["protocol"] == "experiment-005-nested-spatial-cv"
    assert confirmation["primary_family"] == "spatial"
    assert confirmation["criterion_met"] is True
    assert confirmation["wins_vs_global"] == 4
    assert confirmation["residual_reduction_vs_global"] == pytest.approx(0.06)


def test_controlled_rerun_rejects_wrong_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(experiment_008, "outer_slab_positive_precheck", lambda *args: [])
    monkeypatch.setattr(
        experiment_008,
        "run_nested_spatial_cv_damped",
        lambda *args: _evaluation(protocol="changed-protocol"),
    )

    with pytest.raises(RuntimeError, match="frozen Experiment 005 protocol"):
        experiment_008.run_controlled_h01_rerun(
            _sample(),
            np.array([1, 2], dtype=np.int64),
            np.zeros((2, 3), dtype=float),
        )
