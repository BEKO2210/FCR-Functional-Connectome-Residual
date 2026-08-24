from datetime import UTC, datetime

import pandas as pd
import pytest

from fcr.data.microns import MICrONSConfig
from fcr.data.microns_doctor import run_microns_doctor


class FakeMaterialize:
    synapse_table = "synapses_pni_2"

    def __init__(self, *, missing_table: bool = False):
        self.missing_table = missing_table
        self.secret_token = "must-never-appear"
        self.calls = []

    def get_versions(self, expired=False):
        self.calls.append(("get_versions", expired))
        return [1800, 1822]

    def get_tables(self, version=None):
        self.calls.append(("get_tables", version))
        tables = [
            "proofreading_status_and_strategy",
            "aibs_metamodel_celltypes_v661",
            "synapses_pni_2",
        ]
        if self.missing_table:
            tables.remove("aibs_metamodel_celltypes_v661")
        return tables

    def get_version_metadata(self, version=None):
        self.calls.append(("get_version_metadata", version))
        return {"time_stamp": datetime(2026, 8, 1, tzinfo=UTC)}

    def query_table(self, table, **kwargs):
        self.calls.append(("query_table", table, kwargs))
        if table == "proofreading_status_and_strategy":
            return pd.DataFrame(
                {
                    "pt_root_id": [10],
                    "valid_id": [10],
                    "pt_position_x": [1.0],
                    "pt_position_y": [2.0],
                    "pt_position_z": [3.0],
                    "status_axon": ["t"],
                    "status_dendrite": ["t"],
                    "strategy_axon": ["axon_fully_extended"],
                }
            )
        if table == "aibs_metamodel_celltypes_v661":
            return pd.DataFrame({"pt_root_id": [10], "cell_type": ["L2/3 IT"]})
        raise AssertionError(table)

    def synapse_query(self, **kwargs):
        self.calls.append(("synapse_query", kwargs))
        return pd.DataFrame(
            {"pre_pt_root_id": [10], "post_pt_root_id": [20]}
        )


class FakeClient:
    def __init__(self, *, missing_table: bool = False):
        self.version = None
        self.materialize = FakeMaterialize(missing_table=missing_table)
        self.auth_token = "also-must-never-appear"


def test_doctor_runs_bounded_preflight_and_emits_no_credentials():
    client = FakeClient()
    config = MICrONSConfig(materialization_version=1822, max_nodes=2)

    result = run_microns_doctor(config, client=client)

    assert client.version == 1822
    assert result["ok"] is True
    assert result["materialization_version"] == 1822
    assert result["materialization_timestamp"] == "2026-08-01T00:00:00+00:00"
    assert result["proofreading_probe_rows"] == 1
    assert result["cell_type_probe_rows"] == 1
    assert result["synapse_probe_rows"] == 1
    serialized = repr(result).lower()
    assert "must-never-appear" not in serialized
    assert "auth_token" not in serialized
    assert "secret_token" not in serialized

    proof_call = next(
        call
        for call in client.materialize.calls
        if call[0] == "query_table" and call[1] == "proofreading_status_and_strategy"
    )
    assert proof_call[2]["limit"] == 5
    cell_call = next(
        call
        for call in client.materialize.calls
        if call[0] == "query_table" and call[1] == "aibs_metamodel_celltypes_v661"
    )
    assert cell_call[2]["limit"] == 1
    synapse_call = next(call for call in client.materialize.calls if call[0] == "synapse_query")
    assert synapse_call[1]["limit"] == 1


def test_doctor_fails_if_required_table_is_missing():
    client = FakeClient(missing_table=True)
    config = MICrONSConfig(materialization_version=1822, max_nodes=2)
    with pytest.raises(RuntimeError, match="required MICrONS tables"):
        run_microns_doctor(config, client=client)


def test_doctor_fails_if_requested_materialization_is_unavailable():
    client = FakeClient()
    config = MICrONSConfig(materialization_version=1900, max_nodes=2)
    with pytest.raises(RuntimeError, match="not available"):
        run_microns_doctor(config, client=client)
