from __future__ import annotations

import numpy as np

from fcr.nested_cv import (
    contiguous_node_slabs,
    degree_oracle_diagnostic,
    run_nested_spatial_cv,
)
from fcr.schema import ConnectomeSample


def _sample(n_nodes: int = 25) -> tuple[ConnectomeSample, np.ndarray, np.ndarray]:
    node_ids = np.arange(n_nodes, dtype=np.int64)
    xyz = np.column_stack(
        [
            np.arange(n_nodes, dtype=float) * 20.0,
            (np.arange(n_nodes) % 5).astype(float) * 11.0,
            (np.arange(n_nodes) % 4).astype(float) * 13.0,
        ]
    )
    source = np.repeat(node_ids, n_nodes)
    target = np.tile(node_ids, n_nodes)
    keep = source != target
    source = source[keep]
    target = target[keep]
    source_xyz = xyz[source]
    target_xyz = xyz[target]
    distance = np.linalg.norm(source_xyz - target_xyz, axis=1)
    # Smooth deterministic geometry signal with positives in every spatial slab.
    connected = ((distance < 55.0) | ((source + target) % 19 == 0)).astype(np.int8)
    sample = ConnectomeSample(
        source=source,
        target=target,
        source_type=np.full(len(source), "e"),
        target_type=np.full(len(source), "e"),
        distance=distance,
        connected=connected,
        source_xyz=source_xyz,
        target_xyz=target_xyz,
    )
    return sample, node_ids, xyz


def test_contiguous_slabs_are_disjoint_and_cover_all_nodes() -> None:
    _, node_ids, xyz = _sample()
    slabs = contiguous_node_slabs(node_ids, xyz)
    assert len(slabs) == 5
    assert sorted(np.concatenate(slabs).tolist()) == node_ids.tolist()
    for i, left in enumerate(slabs):
        for j, right in enumerate(slabs):
            if i != j:
                assert np.intersect1d(left, right).size == 0


def test_degree_oracle_charges_both_degree_vectors() -> None:
    sample, _, _ = _sample(10)
    result = degree_oracle_diagnostic(sample)
    # n=10 -> ceil(log2(10))=4 bits per degree, two degree vectors.
    assert result.degree_side_bits == 80
    assert result.residual_bits >= 0
    assert result.total_bits == result.residual_bits + result.degree_side_bits


def test_nested_spatial_cv_is_deterministic_and_node_disjoint() -> None:
    sample, node_ids, xyz = _sample()
    first = run_nested_spatial_cv(sample, node_ids, xyz)
    second = run_nested_spatial_cv(sample, node_ids, xyz)
    assert first["primary"] == second["primary"]
    assert first["aggregate"] == second["aggregate"]
    assert len(first["folds"]) == 5

    seen: set[int] = set()
    for fold in first["folds"]:
        ids = set(fold["test_node_ids"])
        assert not seen.intersection(ids)
        seen.update(ids)
        assert fold["test_pairs"] == fold["test_node_count"] * (fold["test_node_count"] - 1)
        assert set(fold["predictive"]) == {"global", "distance", "relative", "spatial"}
        assert fold["inner_selection"]["distance"]["selected_l2"] in {0.1, 1.0, 10.0}
        assert fold["inner_selection"]["relative"]["selected_l2"] in {0.1, 1.0, 10.0}
        assert fold["inner_selection"]["spatial"]["selected_l2"] in {1.0, 10.0, 100.0}
    assert seen == set(node_ids.tolist())
