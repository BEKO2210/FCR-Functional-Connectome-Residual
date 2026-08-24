import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fcr.data.microns import (
    MICrONSConfig,
    query_microns_pilot,
    save_microns_export,
    validate_microns_export,
)
from fcr.data.microns_transform import build_candidate_data


class FakeMaterialize:
    def __init__(self):
        self.calls = []

    def query_table(self, table, **kwargs):
        self.calls.append(("query_table", table, kwargs))
        if table == "proofreading_status_and_strategy":
            return pd.DataFrame(
                {
                    "pt_root_id": [30, 10, 20, 40],
                    "valid_id": [30, 10, 999, 40],
                    "pt_position_x": [30.0, 10.0, 20.0, 40.0],
                    "pt_position_y": [0.0, 0.0, 0.0, 0.0],
                    "pt_position_z": [0.0, 0.0, 0.0, 0.0],
                    "status_axon": ["t", "true", "t", "t"],
                    "status_dendrite": ["t", "1", "t", "f"],
                    "strategy_axon": ["axon_fully_extended"] * 4,
                }
            )
        if table == "aibs_metamodel_celltypes_v661":
            return pd.DataFrame({"pt_root_id": [10, 30], "cell_type": ["A", "C"]})
        raise AssertionError(table)

    def synapse_query(self, **kwargs):
        self.calls.append(("synapse_query", None, kwargs))
        return pd.DataFrame(
            {
                "pre_pt_root_id": [10, 10],
                "post_pt_root_id": [30, 30],
            }
        )


class FakeClient:
    def __init__(self, materialize=None):
        self.version = None
        self.materialize = materialize or FakeMaterialize()


class BatchedMaterialize:
    def __init__(self):
        self.calls = []

    def query_table(self, table, **kwargs):
        self.calls.append(("query_table", table, kwargs))
        if table == "proofreading_status_and_strategy":
            roots = [10, 20, 30, 40, 50, 60]
            return pd.DataFrame(
                {
                    "pt_root_id": roots,
                    "valid_id": roots,
                    "pt_position_x": [float(x) for x in roots],
                    "pt_position_y": [0.0] * len(roots),
                    "pt_position_z": [0.0] * len(roots),
                    "status_axon": ["t"] * len(roots),
                    "status_dendrite": ["t"] * len(roots),
                    "strategy_axon": ["axon_fully_extended"] * len(roots),
                }
            )
        if table == "aibs_metamodel_celltypes_v661":
            ids = kwargs["filter_in_dict"]["pt_root_id"]
            if ids == [10, 20]:
                return pd.DataFrame(
                    {
                        "pt_root_id": [10, 20, 20],
                        "cell_type": ["A", "B", "B-ambiguous"],
                    }
                )
            if ids == [30, 40]:
                return pd.DataFrame(
                    {"pt_root_id": [30, 40], "cell_type": ["C", "D"]}
                )
            if ids == [50, 60]:
                raise AssertionError("selection should stop once max_nodes is satisfied")
            raise AssertionError(ids)
        raise AssertionError(table)

    def synapse_query(self, **kwargs):
        self.calls.append(("synapse_query", None, kwargs))
        return pd.DataFrame(columns=["pre_pt_root_id", "post_pt_root_id"])


def _two_node_export(tmp_path: Path):
    nodes = pd.DataFrame(
        {
            "pt_root_id": [10, 20],
            "pt_position_x": [0.0, 3.0],
            "pt_position_y": [0.0, 4.0],
            "pt_position_z": [0.0, 0.0],
            "cell_type": ["A", "B"],
            "strategy_axon": ["axon_fully_extended", "axon_fully_extended"],
        }
    )
    synapses = pd.DataFrame({"pre_pt_root_id": [10], "post_pt_root_id": [20]})
    data = build_candidate_data(nodes, synapses)
    config = MICrONSConfig(materialization_version=1822, max_nodes=2)
    return save_microns_export(data, nodes, config, tmp_path / "pilot.npz")


def test_config_requires_explicit_positive_materialization_version():
    with pytest.raises(ValueError):
        MICrONSConfig(materialization_version=0)


def test_config_rejects_invalid_cell_type_batch_size():
    with pytest.raises(ValueError):
        MICrONSConfig(materialization_version=1822, cell_type_batch_size=0)


def test_query_sets_version_and_applies_strict_validity_filters():
    client = FakeClient()
    config = MICrONSConfig(materialization_version=1822, max_nodes=10)
    nodes, synapses = query_microns_pilot(config, client=client)
    assert client.version == 1822
    assert nodes["pt_root_id"].tolist() == [10, 30]
    assert len(synapses) == 2
    proof_call = client.materialize.calls[0]
    assert proof_call[2]["materialization_version"] == 1822
    assert proof_call[2]["desired_resolution"] == [1, 1, 1]
    syn_call = client.materialize.calls[-1]
    assert syn_call[2]["pre_ids"] == [10, 30]
    assert syn_call[2]["post_ids"] == [10, 30]
    assert syn_call[2]["remove_autapses"] is True
    assert "materialization_version" not in syn_call[2]


def test_cell_type_batches_continue_until_post_filter_max_nodes():
    materialize = BatchedMaterialize()
    client = FakeClient(materialize)
    config = MICrONSConfig(
        materialization_version=1822,
        max_nodes=3,
        cell_type_batch_size=2,
    )
    nodes, _ = query_microns_pilot(config, client=client)

    assert nodes["pt_root_id"].tolist() == [10, 30, 40]
    assert nodes["cell_type"].tolist() == ["A", "C", "D"]
    cell_type_calls = [
        call
        for call in materialize.calls
        if call[0] == "query_table" and call[1] == "aibs_metamodel_celltypes_v661"
    ]
    assert [call[2]["filter_in_dict"]["pt_root_id"] for call in cell_type_calls] == [
        [10, 20],
        [30, 40],
    ]


def test_export_writes_self_validating_npz_and_provenance_without_token(tmp_path: Path):
    npz_path, sidecar = _two_node_export(tmp_path)

    with np.load(npz_path, allow_pickle=False) as payload:
        assert payload["node_id"].tolist() == [10, 20]
        assert payload["node_xyz_nm"].tolist() == [[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]
        assert payload["connected"].tolist() == [1, 0]
        assert payload["distance_nm"].tolist() == [5.0, 5.0]

    summary = validate_microns_export(npz_path)
    assert summary["valid"] is True
    assert summary["materialization_version"] == 1822
    assert summary["node_count"] == 2
    assert summary["candidate_pair_count"] == 2
    assert summary["connected_pair_count"] == 1
    assert summary["synapse_count_total"] == 1

    text = sidecar.read_text()
    assert '"materialization_version": 1822' in text
    assert '"npz_sha256":' in text
    assert "token" not in text.lower()


def test_validator_rejects_npz_hash_tampering(tmp_path: Path):
    npz_path, _ = _two_node_export(tmp_path)
    with npz_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_microns_export(npz_path)


def test_validator_rejects_strategy_not_declared_in_provenance(tmp_path: Path):
    npz_path, sidecar = _two_node_export(tmp_path)
    provenance = json.loads(sidecar.read_text())
    provenance["config"]["proofread_strategies"] = ["different_strategy"]
    sidecar.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="strategy"):
        validate_microns_export(npz_path)
