"""Outcome-blind DANDI/NWB preflight for Experiment 009.

This module may read structural coregistration identifiers and scalar NWB metadata,
but it must never read calcium, movie, or stimulus-response array values.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DANDISET_ID = "000402"
DANDISET_VERSION = "draft"
FROZEN_ASSET_PATH = "sub-17797/sub-17797_ses-9-scan-4_behavior+image+ophys.nwb"
MAX_EXACT_FLOAT_INTEGER = 2**53

_ALLOWED_VALUE_PATHS = {
    "/identifier",
    "/session_description",
    "/session_start_time",
}
_ALLOWED_PLANE_VALUE_COLUMNS = {"id", "pt_root_id"}


@dataclass(frozen=True)
class RootIdSummary:
    dtype: str
    non_null_count: int
    unique_count: int
    duplicate_rows: int
    precision_safe: bool
    normalized: tuple[int, ...] | None
    valid_indices: tuple[int, ...]


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_scalar(value.item())
    return str(value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _read_allowed_value(dataset: Any, path: str) -> np.ndarray:
    """Read only explicitly whitelisted metadata/coregistration datasets."""
    normalized_path = "/" + path.strip("/")
    basename = normalized_path.rsplit("/", 1)[-1]
    plane_path = "/processing/ophys/ImageSegmentation/PlaneSegmentation" in normalized_path
    if normalized_path not in _ALLOWED_VALUE_PATHS and not (
        plane_path and basename in _ALLOWED_PLANE_VALUE_COLUMNS
    ):
        raise RuntimeError(f"Experiment 009 Stage A forbids value reads from {normalized_path}")
    return np.asarray(dataset[()])


def summarize_root_ids(values: np.ndarray) -> RootIdSummary:
    """Summarize root IDs without silently converting unsafe float64 identifiers."""
    array = np.asarray(values).reshape(-1)
    kind = array.dtype.kind

    if kind in "iu":
        valid_mask = array > 0
        valid_indices = np.flatnonzero(valid_mask)
        normalized = tuple(int(value) for value in array[valid_mask])
        unique_count = len(set(normalized))
        return RootIdSummary(
            dtype=str(array.dtype),
            non_null_count=len(normalized),
            unique_count=unique_count,
            duplicate_rows=len(normalized) - unique_count,
            precision_safe=True,
            normalized=normalized,
            valid_indices=tuple(int(index) for index in valid_indices),
        )

    if kind == "f":
        valid_mask = np.isfinite(array) & (array > 0)
        valid_indices = np.flatnonzero(valid_mask)
        finite = array[valid_mask]
        integral = np.all(finite == np.floor(finite)) if len(finite) else True
        exact_range = np.all(np.abs(finite) <= MAX_EXACT_FLOAT_INTEGER) if len(finite) else True
        precision_safe = bool(integral and exact_range)
        unique_count = int(len(np.unique(finite)))
        normalized = tuple(int(value) for value in finite) if precision_safe else None
        return RootIdSummary(
            dtype=str(array.dtype),
            non_null_count=int(len(finite)),
            unique_count=unique_count,
            duplicate_rows=int(len(finite) - unique_count),
            precision_safe=precision_safe,
            normalized=normalized,
            valid_indices=tuple(int(index) for index in valid_indices),
        )

    normalized_values: list[int] = []
    valid_indices_list: list[int] = []
    for index, raw in enumerate(array):
        text = _decode_scalar(raw).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            continue
        try:
            value = int(text)
        except ValueError:
            return RootIdSummary(
                dtype=str(array.dtype),
                non_null_count=0,
                unique_count=0,
                duplicate_rows=0,
                precision_safe=False,
                normalized=None,
                valid_indices=(),
            )
        if value <= 0:
            continue
        normalized_values.append(value)
        valid_indices_list.append(index)

    unique_count = len(set(normalized_values))
    return RootIdSummary(
        dtype=str(array.dtype),
        non_null_count=len(normalized_values),
        unique_count=unique_count,
        duplicate_rows=len(normalized_values) - unique_count,
        precision_safe=True,
        normalized=tuple(normalized_values),
        valid_indices=tuple(valid_indices_list),
    )


def cohort_csv_bytes(rows: list[dict[str, object]]) -> bytes:
    """Create a deterministic metadata-only structural cohort manifest."""
    ordered = sorted(
        rows,
        key=lambda row: (str(row["plane"]), str(row["roi_id"]), int(row["pt_root_id"])),
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=["asset_path", "plane", "roi_id", "pt_root_id"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(ordered)
    return stream.getvalue().encode("utf-8")


def _dataset_metadata(dataset: Any) -> dict[str, object]:
    return {
        "shape": [int(value) for value in dataset.shape],
        "dtype": str(dataset.dtype),
    }


def _scan_roi_response_series(ophys: Any) -> list[dict[str, object]]:
    """Return shapes/dtypes only. Never index a functional dataset."""
    series: list[dict[str, object]] = []

    def walk(group: Any, path: str) -> None:
        if not hasattr(group, "keys"):
            return
        for name in sorted(group.keys()):
            child = group[name]
            child_path = f"{path}/{name}"
            if hasattr(child, "keys"):
                if "RoiResponseSeries" in name:
                    entry: dict[str, object] = {"path": child_path}
                    for dataset_name in ("data", "timestamps"):
                        if dataset_name in child:
                            entry[dataset_name] = _dataset_metadata(child[dataset_name])
                    series.append(entry)
                else:
                    walk(child, child_path)

    walk(ophys, "/processing/ophys")
    return series


def scan_hdf5_metadata(h5: Any, *, asset_path: str) -> tuple[dict[str, object], bytes | None]:
    """Inspect NWB metadata/coregistration without reading functional values."""
    scalar_metadata: dict[str, str] = {}
    value_read_paths: list[str] = []
    for key in ("identifier", "session_description", "session_start_time"):
        if key in h5:
            path = f"/{key}"
            scalar_metadata[key] = _decode_scalar(_read_allowed_value(h5[key], path))
            value_read_paths.append(path)

    try:
        ophys = h5["processing"]["ophys"]
        image_segmentation = ophys["ImageSegmentation"]
    except KeyError as exc:
        raise RuntimeError("MICrONS NWB ophys/ImageSegmentation structure is missing") from exc

    plane_reports: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []
    all_root_ids_safe = True

    for plane_name in sorted(image_segmentation.keys()):
        if not plane_name.startswith("PlaneSegmentation"):
            continue
        plane = image_segmentation[plane_name]
        columns = sorted(plane.keys())
        row_count = int(plane["id"].shape[0]) if "id" in plane else 0
        report: dict[str, object] = {
            "name": plane_name,
            "row_count": row_count,
            "columns": columns,
            "column_metadata": {
                name: _dataset_metadata(plane[name])
                for name in columns
                if hasattr(plane[name], "shape") and hasattr(plane[name], "dtype")
            },
        }

        if "pt_root_id" not in plane or "id" not in plane:
            report["pt_root_id"] = {"present": False}
            plane_reports.append(report)
            continue

        root_path = f"/processing/ophys/ImageSegmentation/{plane_name}/pt_root_id"
        roi_path = f"/processing/ophys/ImageSegmentation/{plane_name}/id"
        root_values = _read_allowed_value(plane["pt_root_id"], root_path)
        roi_values = _read_allowed_value(plane["id"], roi_path).reshape(-1)
        value_read_paths.extend([root_path, roi_path])
        roots = summarize_root_ids(root_values)
        all_root_ids_safe = all_root_ids_safe and roots.precision_safe
        report["pt_root_id"] = {
            "present": True,
            "dtype": roots.dtype,
            "non_null_count": roots.non_null_count,
            "unique_count": roots.unique_count,
            "duplicate_rows": roots.duplicate_rows,
            "precision_safe_for_exact_integer_identity": roots.precision_safe,
        }

        if roots.precision_safe and roots.normalized is not None:
            for source_index, root_id in zip(
                roots.valid_indices,
                roots.normalized,
                strict=True,
            ):
                cohort_rows.append(
                    {
                        "asset_path": asset_path,
                        "plane": plane_name,
                        "roi_id": _decode_scalar(roi_values[source_index]),
                        "pt_root_id": root_id,
                    }
                )
        plane_reports.append(report)

    response_series = _scan_roi_response_series(ophys)

    interval_reports: list[dict[str, object]] = []
    if "intervals" in h5:
        intervals = h5["intervals"]
        for name in sorted(intervals.keys()):
            table = intervals[name]
            if not hasattr(table, "keys"):
                continue
            columns = sorted(table.keys())
            row_count = int(table["id"].shape[0]) if "id" in table else None
            interval_reports.append(
                {
                    "name": name,
                    "row_count": row_count,
                    "columns": columns,
                }
            )

    manifest_bytes = cohort_csv_bytes(cohort_rows) if all_root_ids_safe and cohort_rows else None
    report = {
        "nwb_scalar_metadata": scalar_metadata,
        "plane_segmentations": plane_reports,
        "roi_response_series_metadata_only": response_series,
        "stimulus_interval_metadata_only": interval_reports,
        "coregistration": {
            "root_id_precision_safe": all_root_ids_safe,
            "candidate_rows": len(cohort_rows) if manifest_bytes is not None else 0,
            "candidate_manifest_sha256": (
                hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes is not None else None
            ),
        },
        "value_read_paths": value_read_paths,
        "functional_values_read": False,
    }
    return report, manifest_bytes


def run_dandi_preflight(output_json: str | Path, output_csv: str | Path) -> dict[str, object]:
    """Stream the frozen public NWB asset and run the Stage-A metadata preflight."""
    try:
        import fsspec
        import h5py
        from dandi.dandiapi import DandiAPIClient
    except ImportError as exc:  # pragma: no cover - exercised only in live workflow
        raise RuntimeError(
            "Experiment 009 live preflight requires dandi, fsspec, and h5py"
        ) from exc

    with DandiAPIClient() as client:
        dandiset = client.get_dandiset(DANDISET_ID, DANDISET_VERSION)
        asset = dandiset.get_asset_by_path(FROZEN_ASSET_PATH)
        raw_metadata = asset.get_raw_metadata()
        content_url = asset.get_content_url(follow_redirects=1, strip_query=True)
        asset_report = {
            "dandiset_id": DANDISET_ID,
            "version": DANDISET_VERSION,
            "asset_path": FROZEN_ASSET_PATH,
            "asset_id": str(asset.identifier),
            "asset_size_bytes": int(asset.size),
            "digest": _json_safe(raw_metadata.get("digest")),
            "content_url_host": content_url.split("/", 3)[2],
        }

    fs = fsspec.filesystem("http")
    with fs.open(content_url, "rb", block_size=8 * 1024 * 1024) as remote:
        with h5py.File(remote, mode="r") as h5:
            nwb_report, manifest_bytes = scan_hdf5_metadata(h5, asset_path=FROZEN_ASSET_PATH)

    report = {
        "experiment": "009",
        "stage": "A-outcome-blind-coregistration-preflight",
        "preregistration_issue": 27,
        "asset": asset_report,
        "nwb": nwb_report,
        "h01_or_connectivity_accessed": False,
        "functional_values_read": False,
        "evidence_level": "E0-plumbing-only",
    }

    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_bytes is not None:
        csv_path.write_bytes(manifest_bytes)
    elif csv_path.exists():
        csv_path.unlink()

    return report
