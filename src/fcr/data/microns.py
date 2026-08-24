"""Version-pinned MICrONS CAVE adapter.

The live adapter intentionally never accepts a token argument. CAVE credentials
must be configured locally with CAVEclient and are never written by FCR.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .microns_transform import MICrONSCandidateData, build_candidate_data, normalize_nodes

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
    return (
        proof.sort_values("pt_root_id", kind="stable")
        .drop_duplicates("pt_root_id")
        .reset_index(drop=True)
    )


def _query_cell_types(client: Any, root_ids: np.ndarray, config: MICrONSConfig) -> pd.DataFrame:
    return client.materialize.query_table(
        config.cell_type_table,
        filter_in_dict={"pt_root_id": [int(x) for x in root_ids]},
        select_columns=["pt_root_id", "cell_type"],
        materialization_version=config.materialization_version,
    )


def _unique_cell_types(cell_types: pd.DataFrame) -> pd.DataFrame:
    """Keep only roots with exactly one automated cell-type row.

    Multiple rows for one root can indicate multiple nuclei/cell bodies. Silently
    choosing the first row would turn an ambiguous biological object into a
    deterministic label, so ambiguous roots are excluded instead.
    """
    required = {"pt_root_id", "cell_type"}
    missing = required - set(cell_types.columns)
    if missing:
        raise RuntimeError(f"cell-type query missing columns: {sorted(missing)}")
    out = cell_types[["pt_root_id", "cell_type"]].dropna(subset=["pt_root_id"]).copy()
    out["pt_root_id"] = out["pt_root_id"].astype(np.int64)
    return out.drop_duplicates("pt_root_id", keep=False).reset_index(drop=True)


def query_microns_pilot(
    config: MICrONSConfig, *, client: Any | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query a small, deterministic induced graph of strict proofread neurons.

    The selection is intentionally a plumbing pilot (sorted root IDs), not a
    preregistered biological sample. It must not be used for a headline result.
    """
    live_client = client if client is not None else create_cave_client(config)
    live_client.version = config.materialization_version

    proofread_nodes = _query_proofread_nodes(live_client, config)
    if len(proofread_nodes) < 2:
        raise RuntimeError("fewer than two eligible proofread neurons were returned")

    cell_types = _query_cell_types(
        live_client, proofread_nodes["pt_root_id"].to_numpy(dtype=np.int64), config
    )
    cell_types = _unique_cell_types(cell_types)
    nodes = proofread_nodes.merge(
        cell_types, on="pt_root_id", how="inner", validate="one_to_one"
    )
    nodes = (
        nodes.sort_values("pt_root_id", kind="stable")
        .head(config.max_nodes)
        .reset_index(drop=True)
    )
    if len(nodes) < 2:
        raise RuntimeError("fewer than two eligible neurons have unambiguous cell-type labels")

    root_ids = nodes["pt_root_id"].astype(np.int64).tolist()
    synapses = live_client.materialize.synapse_query(
        pre_ids=root_ids,
        post_ids=root_ids,
        remove_autapses=True,
    )
    return nodes, synapses.reset_index(drop=True)


def sha256_file(path: str | Path) -> str:
    """Return SHA-256 for an on-disk artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_microns_export(
    data: MICrONSCandidateData,
    nodes: pd.DataFrame,
    config: MICrONSConfig,
    output: str | Path,
) -> tuple[Path, Path]:
    """Write a compact NPZ plus self-verifying provenance sidecar."""
    destination = Path(output)
    if destination.suffix != ".npz":
        destination = destination.with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)

    node_df = normalize_nodes(nodes)
    node_strategy = (
        node_df["strategy_axon"].astype(str).to_numpy()
        if "strategy_axon" in node_df.columns
        else np.full(len(node_df), "unknown", dtype=str)
    )
    sample = data.sample
    np.savez_compressed(
        destination,
        node_id=node_df["pt_root_id"].to_numpy(dtype=np.int64),
        node_type=node_df["cell_type"].to_numpy(dtype=str),
        node_xyz_nm=node_df[["pt_position_x", "pt_position_y", "pt_position_z"]].to_numpy(
            dtype=np.float64
        ),
        node_strategy_axon=node_strategy,
        source=np.asarray(sample.source, dtype=np.int64),
        target=np.asarray(sample.target, dtype=np.int64),
        source_type=np.asarray(sample.source_type, dtype=str),
        target_type=np.asarray(sample.target_type, dtype=str),
        distance_nm=np.asarray(sample.distance, dtype=np.float64),
        connected=np.asarray(sample.connected, dtype=np.int8),
        synapse_count=np.asarray(data.synapse_count, dtype=np.int32),
    )

    artifact_sha256 = sha256_file(destination)
    provenance = {
        "created_utc": datetime.now(UTC).isoformat(),
        "evidence_level": "E0-data-plumbing",
        "config": asdict(config),
        "node_count": int(len(node_df)),
        "candidate_pair_count": int(len(sample)),
        "connected_pair_count": int(np.asarray(sample.connected).sum()),
        "synapse_count_total": int(np.asarray(data.synapse_count).sum()),
        "coordinate_units": "nm",
        "npz_sha256": artifact_sha256,
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


def _require_keys(payload: Any, required: set[str]) -> None:
    missing = required - set(payload.files)
    if missing:
        raise ValueError(f"MICrONS export missing arrays: {sorted(missing)}")


def validate_microns_export(path: str | Path) -> dict[str, Any]:
    """Validate export invariants and provenance hash, returning a compact summary."""
    source_path = Path(path)
    sidecar_path = source_path.with_suffix(".provenance.json")
    if not source_path.is_file():
        raise ValueError(f"export does not exist: {source_path}")
    if not sidecar_path.is_file():
        raise ValueError(f"provenance sidecar does not exist: {sidecar_path}")

    provenance = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected_hash = provenance.get("npz_sha256")
    actual_hash = sha256_file(source_path)
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError("NPZ SHA-256 does not match provenance")

    required = {
        "node_id",
        "node_type",
        "node_xyz_nm",
        "node_strategy_axon",
        "source",
        "target",
        "source_type",
        "target_type",
        "distance_nm",
        "connected",
        "synapse_count",
    }
    with np.load(source_path, allow_pickle=False) as payload:
        _require_keys(payload, required)
        node_id = np.asarray(payload["node_id"], dtype=np.int64)
        node_type = np.asarray(payload["node_type"], dtype=str)
        node_xyz = np.asarray(payload["node_xyz_nm"], dtype=float)
        source = np.asarray(payload["source"], dtype=np.int64)
        target = np.asarray(payload["target"], dtype=np.int64)
        source_type = np.asarray(payload["source_type"], dtype=str)
        target_type = np.asarray(payload["target_type"], dtype=str)
        distance = np.asarray(payload["distance_nm"], dtype=float)
        connected = np.asarray(payload["connected"], dtype=np.int8)
        counts = np.asarray(payload["synapse_count"], dtype=np.int64)

    n_nodes = len(node_id)
    expected_pairs = n_nodes * (n_nodes - 1)
    n_pairs = len(source)
    if n_nodes < 2 or len(np.unique(node_id)) != n_nodes:
        raise ValueError("node IDs must contain at least two unique roots")
    if node_xyz.shape != (n_nodes, 3) or len(node_type) != n_nodes:
        raise ValueError("node metadata dimensions are inconsistent")
    if not np.isfinite(node_xyz).all():
        raise ValueError("node coordinates contain non-finite values")

    pair_arrays = (target, source_type, target_type, distance, connected, counts)
    if any(len(array) != n_pairs for array in pair_arrays):
        raise ValueError("candidate-pair arrays have inconsistent lengths")
    if n_pairs != expected_pairs:
        raise ValueError(f"expected {expected_pairs} directed non-self pairs, found {n_pairs}")
    if np.any(source == target):
        raise ValueError("candidate graph contains autapses")
    if np.any(counts < 0):
        raise ValueError("synapse_count contains negative values")
    if not np.array_equal(connected, (counts > 0).astype(np.int8)):
        raise ValueError("connected is inconsistent with synapse_count")
    if not np.isfinite(distance).all() or np.any(distance < 0):
        raise ValueError("distance_nm contains invalid values")

    order = np.argsort(node_id)
    sorted_ids = node_id[order]
    sorted_xyz = node_xyz[order]
    sorted_types = node_type[order]
    source_index = np.searchsorted(sorted_ids, source)
    target_index = np.searchsorted(sorted_ids, target)
    if (
        np.any(source_index >= n_nodes)
        or np.any(target_index >= n_nodes)
        or not np.array_equal(sorted_ids[source_index], source)
        or not np.array_equal(sorted_ids[target_index], target)
    ):
        raise ValueError("candidate pair references unknown node IDs")

    pair_codes = source_index.astype(np.int64) * n_nodes + target_index.astype(np.int64)
    if len(np.unique(pair_codes)) != expected_pairs:
        raise ValueError("candidate graph has duplicate or missing directed pairs")

    expected_distance = np.linalg.norm(sorted_xyz[source_index] - sorted_xyz[target_index], axis=1)
    if not np.allclose(distance, expected_distance, rtol=1e-10, atol=1e-8):
        raise ValueError("distance_nm is inconsistent with exported node coordinates")
    if not np.array_equal(source_type, sorted_types[source_index]):
        raise ValueError("source_type is inconsistent with node_type")
    if not np.array_equal(target_type, sorted_types[target_index]):
        raise ValueError("target_type is inconsistent with node_type")

    connected_pairs = int(connected.sum())
    total_synapses = int(counts.sum())
    expected_provenance = {
        "node_count": n_nodes,
        "candidate_pair_count": n_pairs,
        "connected_pair_count": connected_pairs,
        "synapse_count_total": total_synapses,
    }
    for key, value in expected_provenance.items():
        if provenance.get(key) != value:
            raise ValueError(f"provenance {key} does not match NPZ")

    return {
        "valid": True,
        "npz_sha256": actual_hash,
        "materialization_version": provenance.get("config", {}).get(
            "materialization_version"
        ),
        "node_count": n_nodes,
        "candidate_pair_count": n_pairs,
        "connected_pair_count": connected_pairs,
        "synapse_count_total": total_synapses,
        "coordinate_units": provenance.get("coordinate_units"),
    }


def export_microns_pilot(config: MICrONSConfig, output: str | Path) -> tuple[Path, Path]:
    nodes, synapses = query_microns_pilot(config)
    candidate_data = build_candidate_data(
        nodes,
        synapses,
        max_candidate_pairs=config.max_candidate_pairs,
    )
    paths = save_microns_export(candidate_data, nodes, config, output)
    validate_microns_export(paths[0])
    return paths
