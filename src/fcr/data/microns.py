"""Version-pinned MICrONS CAVE adapter.

The live adapter intentionally never accepts a token argument. CAVE credentials
must be configured locally with CAVEclient and are never written by FCR.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .microns_transform import MICrONSCandidateData, build_candidate_data

_STRICT_AXON_STRATEGIES = ("axon_fully_extended",)


@dataclass(frozen=True)
class MICrONSConfig:
    materialization_version: int
    datastack: str = "minnie65_public"
    max_nodes: int = 500
    max_candidate_pairs: int = 5_000_000
    proofread_strategies: tuple[str, ...] = _STRICT_AXON_STRATEGIES
    cell_type_table: str = "aibs_metamodel_celltypes_v661"
    require_dendrite_status: bool = True
    require_current_valid_id: bool = True

    def __post_init__(self) -> None:
        if self.materialization_version <= 0:
            raise ValueError("materialization_version must be an explicit positive integer")
        if not self.datastack:
            raise ValueError("datastack must not be empty")
        if not 2 <= self.max_nodes <= 5_000:
            raise ValueError("max_nodes must be between 2 and 5,000")
        if self.max_candidate_pairs < 2:
            raise ValueError("max_candidate_pairs must be >= 2")
        if not self.proofread_strategies:
            raise ValueError("at least one proofreading strategy is required")
        if not self.cell_type_table:
            raise ValueError("cell_type_table must not be empty")


def create_cave_client(config: MICrONSConfig) -> Any:
    """Create an authenticated CAVEclient without ever handling token material."""
    try:
        from caveclient import CAVEclient
    except ImportError:
        raise RuntimeError(
            "MICrONS live access requires the optional dependency: "
            "python -m pip install -e '.[microns]'"
        ) from None

    try:
        client = CAVEclient(config.datastack)
        client.version = config.materialization_version
    except Exception:
        raise RuntimeError(
            "CAVE authentication/version setup failed. Configure your CAVEclient token "
            "locally using the official MICrONS instructions; do not put the token in FCR."
        ) from None
    return client


def _status_true(series: pd.Series) -> pd.Series:
    """Parse CAVE boolean-like status fields without treating "f" as truthy."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    return normalized.isin({"t", "true", "1", "yes"})


def _query_proofread_nodes(client: Any, config: MICrONSConfig) -> pd.DataFrame:
    proof = client.materialize.query_table(
        "proofreading_status_and_strategy",
        filter_in_dict={"strategy_axon": list(config.proofread_strategies)},
        desired_resolution=[1, 1, 1],
        split_positions=True,
        materialization_version=config.materialization_version,
    )
    required = {
        "pt_root_id",
        "pt_position_x",
        "pt_position_y",
        "pt_position_z",
        "status_axon",
        "strategy_axon",
    }
    missing = required - set(proof.columns)
    if missing:
        raise RuntimeError(f"proofreading query missing columns: {sorted(missing)}")

    mask = _status_true(proof["status_axon"])
    if config.require_dendrite_status:
        if "status_dendrite" not in proof.columns:
            raise RuntimeError("proofreading query lacks status_dendrite")
        mask &= _status_true(proof["status_dendrite"])
    if config.require_current_valid_id:
        if "valid_id" not in proof.columns:
            raise RuntimeError("proofreading query lacks valid_id")
        valid = pd.to_numeric(proof["valid_id"], errors="coerce")
        root = pd.to_numeric(proof["pt_root_id"], errors="coerce")
        mask &= valid.eq(root)

    proof = proof.loc[mask].copy()
    proof = proof.dropna(subset=["pt_root_id"])
    proof["pt_root_id"] = proof["pt_root_id"].astype(np.int64)
    proof = proof[proof["pt_root_id"] > 0]
    proof = proof.sort_values("pt_root_id", kind="stable").drop_duplicates("pt_root_id")
    return proof.head(config.max_nodes).reset_index(drop=True)


def _query_cell_types(client: Any, root_ids: np.ndarray, config: MICrONSConfig) -> pd.DataFrame:
    return client.materialize.query_table(
        config.cell_type_table,
        filter_in_dict={"pt_root_id": [int(x) for x in root_ids]},
        select_columns=["pt_root_id", "cell_type"],
        materialization_version=config.materialization_version,
    )


def query_microns_pilot(
    config: MICrONSConfig, *, client: Any | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query a small, deterministic induced graph of strict proofread neurons.

    The selection is intentionally a plumbing pilot (sorted root IDs), not a
    preregistered biological sample. It must not be used for a headline result.
    """
    live_client = client if client is not None else create_cave_client(config)
    live_client.version = config.materialization_version

    nodes = _query_proofread_nodes(live_client, config)
    if len(nodes) < 2:
        raise RuntimeError("fewer than two eligible proofread neurons were returned")

    cell_types = _query_cell_types(
        live_client, nodes["pt_root_id"].to_numpy(dtype=np.int64), config
    )
    if "pt_root_id" not in cell_types.columns or "cell_type" not in cell_types.columns:
        raise RuntimeError("cell-type query must return pt_root_id and cell_type")
    cell_types = cell_types[["pt_root_id", "cell_type"]].drop_duplicates("pt_root_id")
    nodes = nodes.merge(cell_types, on="pt_root_id", how="inner", validate="one_to_one")
    if len(nodes) < 2:
        raise RuntimeError("fewer than two eligible neurons have cell-type labels")

    root_ids = nodes["pt_root_id"].astype(np.int64).tolist()
    synapses = live_client.materialize.synapse_query(
        pre_ids=root_ids,
        post_ids=root_ids,
        remove_autapses=True,
    )
    return nodes.reset_index(drop=True), synapses.reset_index(drop=True)


def save_microns_export(
    data: MICrONSCandidateData,
    nodes: pd.DataFrame,
    config: MICrONSConfig,
    output: str | Path,
) -> tuple[Path, Path]:
    """Write a compact NPZ plus human-readable provenance sidecar."""
    destination = Path(output)
    if destination.suffix != ".npz":
        destination = destination.with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)

    sample = data.sample
    np.savez_compressed(
        destination,
        source=np.asarray(sample.source, dtype=np.int64),
        target=np.asarray(sample.target, dtype=np.int64),
        source_type=np.asarray(sample.source_type, dtype=str),
        target_type=np.asarray(sample.target_type, dtype=str),
        distance_nm=np.asarray(sample.distance, dtype=np.float64),
        connected=np.asarray(sample.connected, dtype=np.int8),
        synapse_count=np.asarray(data.synapse_count, dtype=np.int32),
    )

    provenance = {
        "created_utc": datetime.now(UTC).isoformat(),
        "evidence_level": "E0-data-plumbing",
        "config": asdict(config),
        "node_count": int(len(nodes)),
        "candidate_pair_count": int(len(sample)),
        "connected_pair_count": int(np.asarray(sample.connected).sum()),
        "synapse_count_total": int(np.asarray(data.synapse_count).sum()),
        "coordinate_units": "nm",
        "selection_warning": (
            "Deterministic root-id-limited plumbing pilot; not a preregistered biological sample."
        ),
        "zero_edge_warning": (
            "A zero means no queried synapse was observed for this candidate pair; it is not "
            "proof of a biological non-connection."
        ),
    }
    sidecar = destination.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return destination, sidecar


def export_microns_pilot(config: MICrONSConfig, output: str | Path) -> tuple[Path, Path]:
    nodes, synapses = query_microns_pilot(config)
    candidate_data = build_candidate_data(
        nodes,
        synapses,
        max_candidate_pairs=config.max_candidate_pairs,
    )
    return save_microns_export(candidate_data, nodes, config, output)
