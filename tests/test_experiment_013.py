from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from fcr.experiment_013 import (
    EligibleEdge,
    _assign_distance_bins,
    _mark_quartiles,
    load_verified_tuning,
    matched_random_test,
)


def _edge(
    fold: int,
    source: int,
    target: int,
    distance: float,
    surprise: float,
    source_osi: float,
    target_osi: float,
    same_scan: bool,
) -> EligibleEdge:
    return EligibleEdge(
        fold=fold,
        source=source,
        target=target,
        distance_nm=distance,
        probability=2.0 ** (-surprise),
        surprise_bits=surprise,
        source_scan=1,
        target_scan=1 if same_scan else 2,
        source_osi=source_osi,
        target_osi=target_osi,
        source_osi_p=0.001,
        target_osi_p=0.002,
    )


def test_verified_tuning_reads_only_valid_arrays() -> None:
    raw = pickle.dumps(
        {
            "segment_id": np.array([11, 12, 13], dtype=np.uint64),
            "scan_id": np.array([1, 1, 2], dtype=np.int64),
            "osi": np.array([0.1, 0.5, 0.9]),
            "osi_p": np.array([0.01, 0.2, 0.001]),
            "dsi": np.array([999.0, 999.0, 999.0]),
        },
        protocol=4,
    )
    tuning = load_verified_tuning(raw)
    assert tuning.segment_id.tolist() == [11, 12, 13]
    assert tuning.scan_id.tolist() == [1, 1, 2]
    assert tuning.osi.tolist() == [0.1, 0.5, 0.9]


def test_float_segment_ids_are_forbidden() -> None:
    raw = pickle.dumps(
        {
            "segment_id": np.array([11.0, 12.0]),
            "scan_id": np.array([1, 2]),
            "osi": np.array([0.1, 0.2]),
            "osi_p": np.array([0.1, 0.2]),
        },
        protocol=4,
    )
    with pytest.raises(RuntimeError, match="exact one-dimensional integers"):
        load_verified_tuning(raw)


def test_duplicate_segment_ids_are_hard_stop() -> None:
    raw = pickle.dumps(
        {
            "segment_id": np.array([11, 11], dtype=np.uint64),
            "scan_id": np.array([1, 2]),
            "osi": np.array([0.1, 0.2]),
            "osi_p": np.array([0.1, 0.2]),
        },
        protocol=4,
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        load_verified_tuning(raw)


def test_osi_outside_unit_interval_is_hard_stop() -> None:
    raw = pickle.dumps(
        {
            "segment_id": np.array([11, 12], dtype=np.uint64),
            "scan_id": np.array([1, 2]),
            "osi": np.array([0.1, 1.2]),
            "osi_p": np.array([0.1, 0.2]),
        },
        protocol=4,
    )
    with pytest.raises(RuntimeError, match=r"\[0,1\]"):
        load_verified_tuning(raw)


class _Malicious:
    def __reduce__(self):
        return os.system, ("echo should-not-run",)


def test_restricted_unpickler_blocks_foreign_globals() -> None:
    raw = pickle.dumps(
        {
            "segment_id": np.array([11], dtype=np.uint64),
            "scan_id": np.array([1]),
            "osi": np.array([0.1]),
            "osi_p": np.array([0.1]),
            "payload": _Malicious(),
        },
        protocol=4,
    )
    with pytest.raises(RuntimeError, match="restricted deserialization"):
        load_verified_tuning(raw)


def test_quartiles_are_per_fold_and_deterministic() -> None:
    edges = []
    for fold in range(2):
        for i in range(8):
            edges.append(
                _edge(
                    fold,
                    100 * fold + i,
                    100 * fold + i + 20,
                    float(i + 1),
                    float(i + 1),
                    0.1 * i,
                    0.1 * i + 0.01,
                    same_scan=(i % 2 == 0),
                )
            )
    marked = _mark_quartiles(_assign_distance_bins(edges))
    for fold in range(2):
        fold_edges = [edge for edge in marked if edge.fold == fold]
        assert sum(edge.high for edge in fold_edges) == 2
        assert sum(edge.low for edge in fold_edges) == 2
        assert {edge.surprise_bits for edge in fold_edges if edge.high} == {7.0, 8.0}
        assert {edge.surprise_bits for edge in fold_edges if edge.low} == {1.0, 2.0}


def test_matched_random_test_is_reproducible() -> None:
    edges = []
    for fold in range(3):
        for i in range(20):
            surprise = float(i + 1)
            # Deliberately make high-surprise edges more functionally similar.
            delta = 0.01 if i >= 15 else 0.2
            edges.append(
                _edge(
                    fold,
                    1000 * fold + i,
                    1000 * fold + i + 100,
                    float(i + 1),
                    surprise,
                    0.5,
                    0.5 + delta,
                    same_scan=(i % 2 == 0),
                )
            )
    marked = _mark_quartiles(_assign_distance_bins(edges))
    first = matched_random_test(marked)
    second = matched_random_test(marked)
    assert first == second
    assert first["high_edge_count"] == 15
    assert first["high_mean_osi_difference"] < first["random_median_osi_difference"]
