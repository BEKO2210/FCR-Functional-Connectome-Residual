"""Frozen H01 input adapter for Experiment 006 Stage B2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fcr.schema import ConnectomeSample

from .h01_edges import FROZEN_NODE_COUNT, FROZEN_NODE_SHA256, sha256_file

FROZEN_CANDIDATE_SHA256 = "97d3bbc4864a8dc21d18cc02dd04d6c1860765f6e1d9cdf796199de54f34ae76"
FROZEN_SPARSE_EDGE_SHA256 = "6be7d4fe9259321ac00d5835c59dd98d474f5853cd94b4b72a9bd04c359d724d"
FROZEN_CANDIDATE_ROWS = FROZEN_NODE_COUNT * (FROZEN_NODE_COUNT - 1)
FROZEN_CONNECTED_PAIRS = 4_052
FROZEN_SELECTED_SYNAPSES = 6_190


@dataclass(frozen=True)
class H01Experiment006Data:
    sample: ConnectomeSample
    node_ids: np.ndarray
    coordinates_nm: np.ndarray
    synapse_count: np.ndarray


def load_h01_candidate_data(
    nodes_csv: str | Path,
    candidate_table: str | Path,
    *,
    expected_node_sha256: str,
    expected_candidate_sha256: str,
    expected_node_count: int,
    expected_candidate_rows: int,
    expected_connected_pairs: int,
    expected_synapses: int,
) -> H01Experiment006Data:
    """Load a hash-pinned complete directed candidate graph and pair geometry."""
    nodes_path = Path(nodes_csv)
    candidates_path = Path(candidate_table)
    if sha256_file(nodes_path) != expected_node_sha256:
        raise RuntimeError("H01 Experiment 006 node artifact hash mismatch")
    if sha256_file(candidates_path) != expected_candidate_sha256:
        raise RuntimeError("H01 Experiment 006 candidate artifact hash mismatch")

    nodes = pd.read_csv(nodes_path)
    required_node_columns = {
        "c3_rep_manual",
        "celltype",
        "x_nm",
        "y_nm",
        "z_nm",
    }
    if not required_node_columns.issubset(nodes.columns):
        missing = sorted(required_node_columns.difference(nodes.columns))
        raise RuntimeError(f"H01 node artifact missing columns: {missing}")
    if len(nodes) != expected_node_count:
        raise RuntimeError(f"H01 node count mismatch: {len(nodes)} != {expected_node_count}")
    if nodes["c3_rep_manual"].isna().any() or nodes["celltype"].isna().any():
        raise RuntimeError("H01 node artifact contains null identity or cell type")

    node_ids = nodes["c3_rep_manual"].astype(np.int64).to_numpy()
    if len(np.unique(node_ids)) != len(node_ids):
        raise RuntimeError("H01 node artifact contains duplicate identities")
    coordinates_nm = nodes[["x_nm", "y_nm", "z_nm"]].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(coordinates_nm)):
        raise RuntimeError("H01 node coordinates contain non-finite values")

    candidates = pd.read_csv(
        candidates_path,
        compression="infer",
        dtype={
            "pre_id": "int64",
            "post_id": "int64",
            "synapse_count": "int64",
            "connected": "int8",
        },
    )
    expected_columns = ["pre_id", "post_id", "synapse_count", "connected"]
    if candidates.columns.tolist() != expected_columns:
        raise RuntimeError("H01 candidate artifact has an unexpected schema")
    if len(candidates) != expected_candidate_rows:
        raise RuntimeError(
            f"H01 candidate row mismatch: {len(candidates)} != {expected_candidate_rows}"
        )
    if candidates.duplicated(["pre_id", "post_id"]).any():
        raise RuntimeError("H01 candidate artifact contains duplicate ordered pairs")

    source = candidates["pre_id"].to_numpy(dtype=np.int64, copy=False)
    target = candidates["post_id"].to_numpy(dtype=np.int64, copy=False)
    synapse_count = candidates["synapse_count"].to_numpy(dtype=np.int64, copy=False)
    connected = candidates["connected"].to_numpy(dtype=np.int8, copy=False)
    if np.any(source == target):
        raise RuntimeError("H01 primary candidate artifact contains self-pairs")
    if np.any(synapse_count < 0):
        raise RuntimeError("H01 candidate artifact contains negative synapse multiplicity")
    if not np.all((connected == 0) | (connected == 1)):
        raise RuntimeError("H01 candidate connected column is not binary")
    if not np.array_equal(connected, (synapse_count > 0).astype(np.int8)):
        raise RuntimeError("H01 candidate connectivity disagrees with synapse multiplicity")
    if int(connected.sum()) != expected_connected_pairs:
        raise RuntimeError("H01 connected-pair count disagrees with frozen Stage B1 audit")
    if int(synapse_count.sum()) != expected_synapses:
        raise RuntimeError("H01 synapse count disagrees with frozen Stage B1 audit")

    node_index = pd.Index(node_ids)
    source_index = node_index.get_indexer(source)
    target_index = node_index.get_indexer(target)
    if np.any(source_index < 0) or np.any(target_index < 0):
        raise RuntimeError("H01 candidate artifact references a non-frozen node")

    celltype_categories = sorted(str(value) for value in nodes["celltype"].unique())
    celltype_codes = pd.Categorical(
        nodes["celltype"].astype(str), categories=celltype_categories
    ).codes.astype(np.int16)
    source_xyz = coordinates_nm[source_index]
    target_xyz = coordinates_nm[target_index]
    delta = target_xyz - source_xyz
    distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))

    sample = ConnectomeSample(
        source=source,
        target=target,
        source_type=celltype_codes[source_index],
        target_type=celltype_codes[target_index],
        distance=distance,
        connected=connected,
        source_xyz=source_xyz,
        target_xyz=target_xyz,
    )
    return H01Experiment006Data(
        sample=sample,
        node_ids=node_ids,
        coordinates_nm=coordinates_nm,
        synapse_count=synapse_count,
    )


def load_frozen_h01_experiment_006(
    nodes_csv: str | Path,
    candidate_table: str | Path,
) -> H01Experiment006Data:
    """Load exactly the immutable Stage B1 artifacts used by the primary H01 test."""
    return load_h01_candidate_data(
        nodes_csv,
        candidate_table,
        expected_node_sha256=FROZEN_NODE_SHA256,
        expected_candidate_sha256=FROZEN_CANDIDATE_SHA256,
        expected_node_count=FROZEN_NODE_COUNT,
        expected_candidate_rows=FROZEN_CANDIDATE_ROWS,
        expected_connected_pairs=FROZEN_CONNECTED_PAIRS,
        expected_synapses=FROZEN_SELECTED_SYNAPSES,
    )
