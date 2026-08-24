from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fcr.data.microns import MICrONSConfig, query_microns_pilot, save_microns_export
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
    def __init__(self):
        self.version = None
        self.materialize = FakeMaterialize()


def test_config_requires_explicit_positive_materialization_version():
    with pytest.raises(ValueError):
        MICrONSConfig(materialization_version=0)


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


def test_export_writes_npz_and_provenance_without_token(tmp_path: Path):
    nodes = pd.DataFrame(
        {
            "pt_root_id": [10, 20],
            "pt_position_x": [0.0, 3.0],
            "pt_position_y": [0.0, 4.0],
            "pt_position_z": [0.0, 0.0],
            "cell_type": ["A", "B"],
        }
    )
    synapses = pd.DataFrame({"pre_pt_root_id": [10], "post_pt_root_id": [20]})
    data = build_candidate_data(nodes, synapses)
    config = MICrONSConfig(materialization_version=1822, max_nodes=2)
    npz_path, sidecar = save_microns_export(data, nodes, config, tmp_path / "pilot.npz")

    with np.load(npz_path) as payload:
        assert payload["connected"].tolist() == [1, 0]
        assert payload["distance_nm"].tolist() == [5.0, 5.0]
    text = sidecar.read_text()
    assert '"materialization_version": 1822' in text
    assert "token" not in text.lower()
