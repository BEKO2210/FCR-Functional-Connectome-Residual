"""Outcome-blind metadata helpers for H01 Experiment 006 Stage A.

This module intentionally has no parser for H01 synapse/edge contents. It may
inspect soma metadata and storage-object metadata only, as frozen in Issue #13.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_COORDINATE_TOKENS = {"x", "y", "z"}
_LABEL_TOKENS = ("type", "class", "label", "region")
_ID_TOKENS = ("id", "seg", "root", "neuron", "cell")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_coordinate(name: str) -> bool:
    lowered = name.lower()
    parts = {part for part in lowered.replace("-", "_").split("_") if part}
    return bool(parts & _COORDINATE_TOKENS) or any(
        lowered.endswith(suffix) for suffix in ("_x", "_y", "_z", "x_nm", "y_nm", "z_nm")
    )


def _looks_like_label(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in _LABEL_TOKENS)


def _looks_like_id(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in _ID_TOKENS)


def inspect_soma_csv(path: str | Path) -> dict[str, Any]:
    """Return schema/coordinate/label metadata without connectivity information."""
    source = Path(path)
    frame = pd.read_csv(source)
    report: dict[str, Any] = {
        "path": source.name,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "row_count": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "null_counts": {str(column): int(frame[column].isna().sum()) for column in frame.columns},
        "coordinate_candidates": {},
        "label_candidates": {},
        "id_candidates": {},
    }

    for column in frame.columns:
        name = str(column)
        series = frame[column]
        if _looks_like_coordinate(name):
            numeric = pd.to_numeric(series, errors="coerce")
            finite = numeric[np.isfinite(numeric)]
            report["coordinate_candidates"][name] = {
                "finite_count": int(len(finite)),
                "min": float(finite.min()) if len(finite) else None,
                "max": float(finite.max()) if len(finite) else None,
            }
        if _looks_like_label(name):
            values = series.dropna().astype(str)
            unique = sorted(values.unique().tolist())
            entry: dict[str, Any] = {"unique_count": int(len(unique))}
            if len(unique) <= 50:
                counts = values.value_counts(dropna=False).sort_index()
                entry["value_counts"] = {str(key): int(value) for key, value in counts.items()}
            report["label_candidates"][name] = entry
        if _looks_like_id(name):
            values = series.dropna().astype(str)
            counts = values.value_counts()
            report["id_candidates"][name] = {
                "non_null_count": int(len(values)),
                "unique_count": int(values.nunique()),
                "rows_in_duplicated_ids": int(counts[counts > 1].sum()),
                "duplicated_id_count": int((counts > 1).sum()),
            }
    return report


def summarize_object_metadata(
    items: list[dict[str, Any]], *, prefix: str
) -> dict[str, Any]:
    """Summarize cloud object names/sizes only; never inspect object contents."""
    normalized: list[tuple[str, int]] = []
    for item in items:
        name = str(item.get("name", ""))
        if not name.startswith(prefix):
            continue
        size = int(item.get("size", 0))
        normalized.append((name, size))
    sizes = [size for _, size in normalized]
    return {
        "prefix": prefix,
        "object_count": len(normalized),
        "total_bytes": int(sum(sizes)),
        "min_object_bytes": int(min(sizes)) if sizes else None,
        "max_object_bytes": int(max(sizes)) if sizes else None,
        "objects": [
            {"name": name, "bytes": size}
            for name, size in sorted(normalized, key=lambda item: item[0])
        ],
    }


def choose_connectivity_source(
    crest_bytes: int | None, *, crest_limit_bytes: int = 2 * 1024**3
) -> str:
    """Choose source solely from storage feasibility, never from edge outcomes."""
    if crest_bytes is not None and 0 < crest_bytes <= crest_limit_bytes:
        return "official-crest-sqlite"
    return "official-h01-sharded-json"
