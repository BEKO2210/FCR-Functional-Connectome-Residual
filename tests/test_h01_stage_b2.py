from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fcr.data.h01_edges import sha256_file
from fcr.data.h01_experiment_006 import load_h01_candidate_data
from fcr.experiment_006 import outer_slab_positive_precheck, strict_spatial_confirmation
from fcr.schema import ConnectomeSample


def _write_small_h01_fixture(tmp_path: Path) -> tuple[Path, Path]:
    nodes_path = tmp_path / "nodes.csv"
    candidates_path = tmp_path / "candidates.csv.gz"
    nodes = pd.DataFrame(
        {
            "c3_rep_manual": [10, 20, 30],
            "celltype": ["A", "B", "A"],
            "x_nm": [0, 3, 0],
            "y_nm": [0, 4, 0],
            "z_nm": [0, 0, 12],
        }
    )
    nodes.to_csv(nodes_path, index=False, lineterminator="\n")
    candidates = pd.DataFrame(
        {
            "pre_id": [10, 10, 20, 20, 30, 30],
            "post_id": [20, 30, 10, 30, 10, 20],
            "synapse_count": [2, 0, 0, 1, 0, 0],
            "connected": [1, 0, 0, 1, 0, 0],
        }
    )
    candidates.to_csv(
        candidates_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
        lineterminator="\n",
    )
    return nodes_path, candidates_path


def test_h01_adapter_reconstructs_geometry_and_connectivity(tmp_path: Path) -> None:
    nodes_path, candidates_path = _write_small_h01_fixture(tmp_path)
    data = load_h01_candidate_data(
        nodes_path,
        candidates_path,
        expected_node_sha256=sha256_file(nodes_path),
        expected_candidate_sha256=sha256_file(candidates_path),
        expected_node_count=3,
        expected_candidate_rows=6,
        expected_connected_pairs=2,
        expected_synapses=3,
    )

    assert data.node_ids.tolist() == [10, 20, 30]
    assert len(data.sample) == 6
    assert int(data.sample.connected.sum()) == 2
    assert int(data.synapse_count.sum()) == 3
    assert data.sample.distance[0] == pytest.approx(5.0)
    assert data.sample.distance[1] == pytest.approx(12.0)


def test_h01_adapter_rejects_hash_mismatch(tmp_path: Path) -> None:
    nodes_path, candidates_path = _write_small_h01_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="candidate artifact hash mismatch"):
        load_h01_candidate_data(
            nodes_path,
            candidates_path,
            expected_node_sha256=sha256_file(nodes_path),
            expected_candidate_sha256="0" * 64,
            expected_node_count=3,
            expected_candidate_rows=6,
            expected_connected_pairs=2,
            expected_synapses=3,
        )


def _complete_sample_with_five_spatial_positives() -> tuple[ConnectomeSample, np.ndarray, np.ndarray]:
    node_ids = np.arange(15, dtype=np.int64)
    coordinates = np.column_stack(
        [node_ids.astype(float), np.zeros(15), np.zeros(15)]
    )
    source: list[int] = []
    target: list[int] = []
    connected: list[int] = []
    positives = {(0, 1), (3, 4), (6, 7), (9, 10), (12, 13)}
    for pre in node_ids:
        for post in node_ids:
            if pre == post:
                continue
            source.append(int(pre))
            target.append(int(post))
            connected.append(int((int(pre), int(post)) in positives))
    source_array = np.asarray(source, dtype=np.int64)
    target_array = np.asarray(target, dtype=np.int64)
    source_xyz = coordinates[source_array]
    target_xyz = coordinates[target_array]
    distance = np.linalg.norm(target_xyz - source_xyz, axis=1)
    sample = ConnectomeSample(
        source=source_array,
        target=target_array,
        source_type=np.zeros(len(source_array), dtype=np.int8),
        target_type=np.zeros(len(source_array), dtype=np.int8),
        distance=distance,
        connected=np.asarray(connected, dtype=np.int8),
        source_xyz=source_xyz,
        target_xyz=target_xyz,
    )
    return sample, node_ids, coordinates


def test_outer_slab_precheck_stops_on_zero_positive_fold() -> None:
    sample, node_ids, coordinates = _complete_sample_with_five_spatial_positives()
    results = outer_slab_positive_precheck(sample, node_ids, coordinates)
    assert [item["positive_pairs"] for item in results] == [1, 1, 1, 1, 1]

    connected = np.asarray(sample.connected).copy()
    connected[(sample.source == 12) & (sample.target == 13)] = 0
    broken = ConnectomeSample(
        source=sample.source,
        target=sample.target,
        source_type=sample.source_type,
        target_type=sample.target_type,
        distance=sample.distance,
        connected=connected,
        source_xyz=sample.source_xyz,
        target_xyz=sample.target_xyz,
    )
    with pytest.raises(RuntimeError, match="outer slab 4 has zero positive pairs"):
        outer_slab_positive_precheck(broken, node_ids, coordinates)


def test_strict_confirmation_never_substitutes_another_family() -> None:
    evaluation = {
        "aggregate": {
            "global": {"residual_bits": 1000.0, "two_part_bits": 1010.0},
            "distance": {
                "residual_bits": 930.0,
                "two_part_bits": 950.0,
                "residual_reduction_vs_global": 0.07,
                "wins_vs_global": 5,
            },
            "relative": {
                "residual_bits": 900.0,
                "two_part_bits": 930.0,
                "residual_reduction_vs_global": 0.10,
                "wins_vs_global": 5,
            },
            "spatial": {
                "residual_bits": 970.0,
                "two_part_bits": 1020.0,
                "residual_reduction_vs_global": 0.03,
                "wins_vs_global": 5,
            },
        }
    }
    result = strict_spatial_confirmation(evaluation)
    assert result["primary_family"] == "spatial"
    assert result["criterion_met"] is False
    assert result["residual_reduction_vs_global"] == pytest.approx(0.03)
