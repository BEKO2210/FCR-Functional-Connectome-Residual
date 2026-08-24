"""Pure transformations from MICrONS-like tables into FCR candidate pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..schema import ConnectomeSample

_NODE_ID = "pt_root_id"
_POSITION_COLUMNS = ("pt_position_x", "pt_position_y", "pt_position_z")
_PRE = "pre_pt_root_id"
_POST = "post_pt_root_id"


@dataclass(frozen=True)
class MICrONSCandidateData:
    """Binary FCR sample plus synapse multiplicity for the same candidate pairs."""

    sample: ConnectomeSample
    synapse_count: np.ndarray

    def __post_init__(self) -> None:
        counts = np.asarray(self.synapse_count)
        if counts.ndim != 1 or len(counts) != len(self.sample):
            raise ValueError("synapse_count must be one-dimensional and match sample")
        if np.any(counts < 0):
            raise ValueError("synapse_count must be non-negative")
        if not np.array_equal((counts > 0).astype(np.int8), self.sample.connected):
            raise ValueError("connected must equal synapse_count > 0")


def normalize_nodes(nodes: pd.DataFrame, *, type_column: str = "cell_type") -> pd.DataFrame:
    """Validate and normalize the minimal node table used by FCR.

    Position columns are assumed to already be in nanometers. The live MICrONS
    adapter enforces this by querying CAVE with desired_resolution=[1, 1, 1].
    """
    required = {_NODE_ID, *_POSITION_COLUMNS}
    missing = required - set(nodes.columns)
    if missing:
        raise ValueError(f"node table missing required columns: {sorted(missing)}")

    out = nodes.copy()
    if out[_NODE_ID].isna().any():
        raise ValueError("pt_root_id contains missing values")
    out[_NODE_ID] = out[_NODE_ID].astype(np.int64)
    if (out[_NODE_ID] <= 0).any():
        raise ValueError("pt_root_id values must be positive")
    if out[_NODE_ID].duplicated().any():
        raise ValueError("pt_root_id values must be unique")

    for column in _POSITION_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="raise").astype(float)
        if not np.isfinite(out[column].to_numpy()).all():
            raise ValueError(f"{column} must contain only finite values")

    if type_column not in out.columns:
        out[type_column] = "unknown"
    out[type_column] = out[type_column].fillna("unknown").astype(str)
    return out.sort_values(_NODE_ID, kind="stable").reset_index(drop=True)


def aggregate_synapses(synapses: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual MICrONS synapse rows to directed cell-pair counts."""
    required = {_PRE, _POST}
    missing = required - set(synapses.columns)
    if missing:
        raise ValueError(f"synapse table missing required columns: {sorted(missing)}")
    if synapses.empty:
        return pd.DataFrame(columns=[_PRE, _POST, "synapse_count"])

    edges = synapses[[_PRE, _POST]].copy()
    if edges.isna().any().any():
        raise ValueError("synapse root ids must not be missing")
    edges[_PRE] = edges[_PRE].astype(np.int64)
    edges[_POST] = edges[_POST].astype(np.int64)
    edges = edges[(edges[_PRE] > 0) & (edges[_POST] > 0)]
    edges = edges[edges[_PRE] != edges[_POST]]
    if edges.empty:
        return pd.DataFrame(columns=[_PRE, _POST, "synapse_count"])

    return (
        edges.groupby([_PRE, _POST], sort=True, observed=True)
        .size()
        .rename("synapse_count")
        .reset_index()
    )


def build_candidate_data(
    nodes: pd.DataFrame,
    synapses: pd.DataFrame,
    *,
    type_column: str = "cell_type",
    max_candidate_pairs: int = 5_000_000,
) -> MICrONSCandidateData:
    """Build the complete directed, non-self candidate graph over selected nodes.

    Every ordered pair of selected neurons is a candidate. A pair is marked
    connected iff one or more queried synapses are present. This is an
    *observation model*, not proof that every zero is a biological non-edge.
    """
    node_df = normalize_nodes(nodes, type_column=type_column)
    n_nodes = len(node_df)
    if n_nodes < 2:
        raise ValueError("at least two nodes are required")
    n_pairs = n_nodes * (n_nodes - 1)
    if n_pairs > max_candidate_pairs:
        raise ValueError(
            f"candidate graph would contain {n_pairs:,} pairs; "
            f"limit is {max_candidate_pairs:,}"
        )

    ids = node_df[_NODE_ID].to_numpy(dtype=np.int64)
    types = node_df[type_column].to_numpy(dtype=str)
    xyz = node_df[list(_POSITION_COLUMNS)].to_numpy(dtype=float)

    source_index = np.repeat(np.arange(n_nodes), n_nodes)
    target_index = np.tile(np.arange(n_nodes), n_nodes)
    keep = source_index != target_index
    source_index = source_index[keep]
    target_index = target_index[keep]

    source = ids[source_index]
    target = ids[target_index]
    source_type = types[source_index]
    target_type = types[target_index]
    distance = np.linalg.norm(xyz[source_index] - xyz[target_index], axis=1)

    counts_df = aggregate_synapses(synapses)
    selected = set(ids.tolist())
    if not counts_df.empty:
        counts_df = counts_df[
            counts_df[_PRE].isin(selected) & counts_df[_POST].isin(selected)
        ]

    pair_index = pd.MultiIndex.from_arrays([source, target], names=[_PRE, _POST])
    if counts_df.empty:
        synapse_count = np.zeros(n_pairs, dtype=np.int32)
    else:
        count_series = counts_df.set_index([_PRE, _POST])["synapse_count"]
        synapse_count = (
            count_series.reindex(pair_index, fill_value=0).to_numpy(dtype=np.int32)
        )

    connected = (synapse_count > 0).astype(np.int8)
    return MICrONSCandidateData(
        sample=ConnectomeSample(
            source=source,
            target=target,
            source_type=source_type,
            target_type=target_type,
            distance=distance,
            connected=connected,
        ),
        synapse_count=synapse_count,
    )
