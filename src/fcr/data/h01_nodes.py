"""Metadata-only H01 node selection for Experiment 006 Stage B0.

This module intentionally contains no H01 synapse or edge parser. It freezes the
eligible human-neuron population and the central 1,500-node confirmation subset
using only the canonical H01 soma table, as preregistered in Issue #13.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

NEURON_CELLTYPES = frozenset(
    {
        "PYRAMIDAL",
        "INTERNEURON",
        "SPINY_ATYPICAL",
        "SPINY_STELLATE",
        "UNCLASSIFIED_NEURON",
    }
)
C3_ID_COLUMN = "c3_rep_manual"
XYZ_COLUMNS = ("x", "y", "z")
VOXEL_NM = np.asarray([8, 8, 33], dtype=np.int64)
EXPECTED_ELIGIBLE_NEURONS = 15_730
PRIMARY_NODE_COUNT = 1_500


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame) -> None:
    required = {C3_ID_COLUMN, "celltype", "layer", *XYZ_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"H01 soma table missing required columns: {sorted(missing)}")


def eligible_h01_neurons(frame: pd.DataFrame) -> pd.DataFrame:
    """Return neuron-labeled, finite, single-soma C3 identities only.

    Single-soma uniqueness is evaluated across the complete soma table rather
    than within neuron labels only, so any C3 identity associated with multiple
    soma annotations is excluded as a possible merge error.
    """
    _require_columns(frame)
    work = frame[[C3_ID_COLUMN, "celltype", "layer", *XYZ_COLUMNS]].copy()
    work[C3_ID_COLUMN] = pd.to_numeric(work[C3_ID_COLUMN], errors="coerce")
    for column in XYZ_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    finite_identity = work[C3_ID_COLUMN].notna() & np.isfinite(work[C3_ID_COLUMN])
    identity_values = work.loc[finite_identity, C3_ID_COLUMN]
    identity_counts = identity_values.value_counts(dropna=False)
    single_soma_ids = set(identity_counts[identity_counts == 1].index.tolist())

    celltypes = work["celltype"].fillna("").astype(str).str.strip().str.upper()
    finite_xyz = np.ones(len(work), dtype=bool)
    for column in XYZ_COLUMNS:
        finite_xyz &= work[column].notna().to_numpy()
        finite_xyz &= np.isfinite(work[column].to_numpy(dtype=float))

    mask = (
        finite_identity.to_numpy()
        & finite_xyz
        & celltypes.isin(NEURON_CELLTYPES).to_numpy()
        & work[C3_ID_COLUMN].isin(single_soma_ids).to_numpy()
    )
    selected = work.loc[mask].copy()
    selected["celltype"] = celltypes.loc[mask].to_numpy()

    c3_values = selected[C3_ID_COLUMN].to_numpy(dtype=float)
    if not np.all(c3_values == np.floor(c3_values)):
        raise ValueError("c3_rep_manual contains non-integer identities")
    selected[C3_ID_COLUMN] = c3_values.astype(np.int64)

    voxel_xyz = selected[list(XYZ_COLUMNS)].to_numpy(dtype=np.int64)
    xyz_nm = voxel_xyz * VOXEL_NM[None, :]
    selected["x_nm"] = xyz_nm[:, 0]
    selected["y_nm"] = xyz_nm[:, 1]
    selected["z_nm"] = xyz_nm[:, 2]

    selected = selected.sort_values(["x", C3_ID_COLUMN], kind="stable").reset_index(drop=True)
    return selected[
        [C3_ID_COLUMN, "celltype", "layer", "x", "y", "z", "x_nm", "y_nm", "z_nm"]
    ]


def central_rank_window(eligible: pd.DataFrame, count: int = PRIMARY_NODE_COUNT) -> pd.DataFrame:
    """Select the preregistered central rank window without connectivity input."""
    if count <= 0:
        raise ValueError("count must be positive")
    if len(eligible) < count:
        return eligible.copy().reset_index(drop=True)
    start = (len(eligible) - count) // 2
    stop = start + count
    return eligible.iloc[start:stop].copy().reset_index(drop=True)


def freeze_h01_nodes(
    soma_csv: str | Path,
    output_csv: str | Path,
    *,
    expected_eligible: int = EXPECTED_ELIGIBLE_NEURONS,
    primary_count: int = PRIMARY_NODE_COUNT,
) -> dict[str, object]:
    """Freeze the metadata-only H01 node set and return an auditable report."""
    source = Path(soma_csv)
    frame = pd.read_csv(source)
    eligible = eligible_h01_neurons(frame)
    if len(eligible) != expected_eligible:
        raise RuntimeError(
            f"eligible H01 neuron count mismatch: observed={len(eligible)} expected={expected_eligible}"
        )

    primary = central_rank_window(eligible, count=primary_count)
    if len(primary) != min(primary_count, len(eligible)):
        raise RuntimeError("central rank selection produced an unexpected node count")
    if primary[C3_ID_COLUMN].duplicated().any():
        raise RuntimeError("primary H01 node set contains duplicate C3 identities")

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    primary.to_csv(destination, index=False, lineterminator="\n")

    type_counts = primary["celltype"].value_counts().sort_index()
    layer_counts = primary["layer"].fillna("<NA>").astype(str).value_counts().sort_index()
    return {
        "soma_sha256": sha256_file(source),
        "eligible_neurons": int(len(eligible)),
        "primary_nodes": int(len(primary)),
        "c3_id_column": C3_ID_COLUMN,
        "neuron_celltypes": sorted(NEURON_CELLTYPES),
        "coordinate_columns": list(XYZ_COLUMNS),
        "voxel_nm": VOXEL_NM.tolist(),
        "selection": "sort by x then c3_rep_manual; take central rank window",
        "selected_csv_sha256": sha256_file(destination),
        "selected_csv_bytes": destination.stat().st_size,
        "selected_c3_min": int(primary[C3_ID_COLUMN].min()),
        "selected_c3_max": int(primary[C3_ID_COLUMN].max()),
        "selected_x_voxel_min": int(primary["x"].min()),
        "selected_x_voxel_max": int(primary["x"].max()),
        "selected_celltype_counts": {str(k): int(v) for k, v in type_counts.items()},
        "selected_layer_counts": {str(k): int(v) for k, v in layer_counts.items()},
        "connectivity_accessed": False,
    }
