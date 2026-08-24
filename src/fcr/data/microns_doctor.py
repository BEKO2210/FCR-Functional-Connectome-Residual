"""Low-cost preflight checks for the live MICrONS CAVE path."""

from __future__ import annotations

from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Any

import pandas as pd

from .microns import MICrONSConfig, create_cave_client

_PROOF_TABLE = "proofreading_status_and_strategy"


def _installed_version(distribution: str) -> str | None:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None


def _iso_or_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{label} probe missing columns: {sorted(missing)}")


def run_microns_doctor(
    config: MICrONSConfig, *, client: Any | None = None
) -> dict[str, Any]:
    """Run bounded read-only checks before a real MICrONS export.

    The returned dictionary is intentionally credential-free. The function never
    reads or serializes CAVE authentication state.
    """
    live_client = client if client is not None else create_cave_client(config)
    live_client.version = config.materialization_version
    materialize = live_client.materialize

    versions_raw = materialize.get_versions(expired=True)
    versions = {int(item) for item in versions_raw}
    if config.materialization_version not in versions:
        raise RuntimeError(
            f"materialization version {config.materialization_version} is not available"
        )

    tables = set(materialize.get_tables(version=config.materialization_version))
    required_tables = {_PROOF_TABLE, config.cell_type_table}
    synapse_table = getattr(materialize, "synapse_table", None)
    if synapse_table:
        required_tables.add(str(synapse_table))
    missing_tables = required_tables - tables
    if missing_tables:
        raise RuntimeError(f"required MICrONS tables are missing: {sorted(missing_tables)}")

    metadata = materialize.get_version_metadata(version=config.materialization_version)

    proof = materialize.query_table(
        _PROOF_TABLE,
        filter_in_dict={"strategy_axon": list(config.proofread_strategies)},
        desired_resolution=[1, 1, 1],
        split_positions=True,
        materialization_version=config.materialization_version,
        limit=5,
    )
    proof_columns = {
        "pt_root_id",
        "pt_position_x",
        "pt_position_y",
        "pt_position_z",
        "status_axon",
        "strategy_axon",
    }
    if config.require_dendrite_status:
        proof_columns.add("status_dendrite")
    if config.require_current_valid_id:
        proof_columns.add("valid_id")
    _require_columns(proof, proof_columns, "proofreading")
    if proof.empty:
        raise RuntimeError("proofreading probe returned no rows for the configured strategy")

    cell_types = materialize.query_table(
        config.cell_type_table,
        select_columns=["pt_root_id", "cell_type"],
        materialization_version=config.materialization_version,
        limit=1,
    )
    _require_columns(cell_types, {"pt_root_id", "cell_type"}, "cell-type")
    if cell_types.empty:
        raise RuntimeError("cell-type probe returned no rows")

    probe_root = int(pd.to_numeric(proof["pt_root_id"], errors="raise").iloc[0])
    synapses = materialize.synapse_query(
        pre_ids=[probe_root],
        remove_autapses=True,
        limit=1,
    )
    _require_columns(synapses, {"pre_pt_root_id", "post_pt_root_id"}, "synapse")

    return {
        "ok": True,
        "datastack": config.datastack,
        "materialization_version": config.materialization_version,
        "materialization_timestamp": _iso_or_string(metadata.get("time_stamp")),
        "caveclient_version": _installed_version("caveclient"),
        "required_tables": sorted(required_tables),
        "proofreading_probe_rows": int(len(proof)),
        "cell_type_probe_rows": int(len(cell_types)),
        "synapse_probe_rows": int(len(synapses)),
    }
