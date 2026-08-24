"""Token-free adapter for the public MICrONS layer-2/3 v185 tables.

This adapter consumes two small static Zenodo tables:
- ``soma_valence_v185.csv`` for soma coordinates / coarse cell class;
- ``soma_subgraph_synapses_spines_v185.csv`` for the proofread soma-subgraph synapses.

It intentionally does not use CAVE, authentication, dynamic segmentation, or current
materializations. This is an E0/E1 structural pilot on an older public release, not a
replacement for the preregistered current-MICrONS experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import ConnectomeSample


@dataclass(frozen=True)
class PublicL23Data:
    sample: ConnectomeSample
    node_ids: np.ndarray
    coordinates_nm: np.ndarray
    synapse_count: np.ndarray


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _find_endpoint_column(frame: pd.DataFrame, prefix: str) -> str:
    """Find a pre/post root-ID column across historical MICrONS naming variants."""
    explicit = [
        f"{prefix}_pt_root_id",
        f"{prefix}_root_id",
        f"{prefix}_seg_id",
        f"{prefix}_segment_id",
    ]
    for name in explicit:
        if name in frame.columns:
            return name

    prefix_lower = prefix.lower()
    candidates = [
        str(column)
        for column in frame.columns
        if prefix_lower in str(column).lower()
        and ("root" in str(column).lower() or "seg" in str(column).lower())
        and "id" in str(column).lower()
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"could not uniquely identify {prefix} endpoint column; candidates={candidates}"
        )
    return candidates[0]


def _normalise_cell_type(series: pd.Series) -> np.ndarray:
    values = series.fillna("unknown").astype(str).str.strip().str.lower()
    mapping = {
        "e": "excitatory",
        "i": "inhibitory",
        "g": "glia",
    }
    return values.map(lambda item: mapping.get(item, item or "unknown")).to_numpy(dtype=str)


def build_public_l23_candidate_data(
    soma: pd.DataFrame,
    synapses: pd.DataFrame,
    *,
    max_nodes: int | None = None,
) -> PublicL23Data:
    """Build a complete directed candidate graph from the public proofread subgraph.

    Only nodes that occur as endpoints in the supplied proofread-synapse table and have
    finite soma coordinates are retained. Every ordered non-self pair among those nodes
    is represented exactly once; absent observed synapses are encoded as zero.
    """
    _require_columns(
        soma,
        {"pt_root_id", "cell_type", "soma_x_nm", "soma_y_nm", "soma_z_nm"},
        "soma table",
    )
    pre_col = _find_endpoint_column(synapses, "pre")
    post_col = _find_endpoint_column(synapses, "post")

    node_frame = soma[
        ["pt_root_id", "cell_type", "soma_x_nm", "soma_y_nm", "soma_z_nm"]
    ].copy()
    node_frame["pt_root_id"] = pd.to_numeric(node_frame["pt_root_id"], errors="coerce")
    for column in ("soma_x_nm", "soma_y_nm", "soma_z_nm"):
        node_frame[column] = pd.to_numeric(node_frame[column], errors="coerce")
    node_frame = node_frame.dropna(
        subset=["pt_root_id", "soma_x_nm", "soma_y_nm", "soma_z_nm"]
    )
    node_frame["pt_root_id"] = node_frame["pt_root_id"].astype(np.int64)
    node_frame = node_frame.drop_duplicates("pt_root_id", keep=False)

    edges = synapses[[pre_col, post_col]].copy()
    edges[pre_col] = pd.to_numeric(edges[pre_col], errors="coerce")
    edges[post_col] = pd.to_numeric(edges[post_col], errors="coerce")
    edges = edges.dropna(subset=[pre_col, post_col])
    edges[pre_col] = edges[pre_col].astype(np.int64)
    edges[post_col] = edges[post_col].astype(np.int64)
    edges = edges[edges[pre_col] != edges[post_col]]

    endpoint_ids = np.union1d(edges[pre_col].unique(), edges[post_col].unique())
    node_frame = node_frame[node_frame["pt_root_id"].isin(endpoint_ids)].copy()
    node_frame = node_frame.sort_values("pt_root_id", kind="stable").reset_index(drop=True)
    if max_nodes is not None:
        if max_nodes < 3:
            raise ValueError("max_nodes must be at least 3")
        node_frame = node_frame.head(max_nodes).copy()

    if len(node_frame) < 3:
        raise ValueError("fewer than three usable nodes remain after filtering")

    node_ids = node_frame["pt_root_id"].to_numpy(dtype=np.int64)
    allowed = set(int(value) for value in node_ids)
    edges = edges[edges[pre_col].isin(allowed) & edges[post_col].isin(allowed)]
    edge_counts = edges.groupby([pre_col, post_col], sort=False).size().to_dict()

    n_nodes = len(node_ids)
    source = np.repeat(node_ids, n_nodes)
    target = np.tile(node_ids, n_nodes)
    non_self = source != target
    source = source[non_self]
    target = target[non_self]

    node_types = dict(
        zip(node_ids.tolist(), _normalise_cell_type(node_frame["cell_type"]).tolist(), strict=True)
    )
    xyz_lookup = {
        int(row.pt_root_id): np.asarray([row.soma_x_nm, row.soma_y_nm, row.soma_z_nm], dtype=float)
        for row in node_frame.itertuples(index=False)
    }
    source_xyz = np.vstack([xyz_lookup[int(value)] for value in source])
    target_xyz = np.vstack([xyz_lookup[int(value)] for value in target])
    distance = np.linalg.norm(source_xyz - target_xyz, axis=1)
    counts = np.fromiter(
        (int(edge_counts.get((int(a), int(b)), 0)) for a, b in zip(source, target, strict=True)),
        dtype=np.int64,
        count=len(source),
    )

    if np.any(distance <= 0):
        raise ValueError("distinct exported nodes must have positive soma distance")

    sample = ConnectomeSample(
        source=source,
        target=target,
        source_type=np.asarray([node_types[int(value)] for value in source], dtype=str),
        target_type=np.asarray([node_types[int(value)] for value in target], dtype=str),
        distance=distance,
        connected=(counts > 0).astype(np.int8),
    )
    coordinates = node_frame[["soma_x_nm", "soma_y_nm", "soma_z_nm"]].to_numpy(dtype=float)
    return PublicL23Data(
        sample=sample,
        node_ids=node_ids,
        coordinates_nm=coordinates,
        synapse_count=counts,
    )


def load_public_l23_candidate_data(
    soma_csv: str | Path,
    synapse_csv: str | Path,
    *,
    max_nodes: int | None = None,
) -> PublicL23Data:
    soma = pd.read_csv(soma_csv)
    synapses = pd.read_csv(synapse_csv)
    return build_public_l23_candidate_data(soma, synapses, max_nodes=max_nodes)
