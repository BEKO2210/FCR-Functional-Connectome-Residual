"""Outcome-blind DANDI/NWB preflight for Experiment 009.

This module may read structural coregistration identifiers and scalar NWB metadata,
but it must never read calcium, movie, or stimulus-response array values.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DANDISET_ID = "000402"
DANDISET_VERSION = "draft"
FROZEN_ASSET_PATH = "sub-17797/sub-17797_ses-9-scan-4_behavior+image+ophys.nwb"
NUCLEUS_DETECTION_URL = (
    "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/"
    "nucleus_detection/nucleus_detection_v0.csv"
)
MAX_EXACT_FLOAT_INTEGER = 2**53
COREG_SOURCE = "manual_match_cave_nuclei_id"

_ALLOWED_VALUE_PATHS = {
    "/identifier",
    "/session_description",
    "/session_start_time",
}
_ALLOWED_PLANE_VALUE_COLUMNS = {
    "id",
    "pt_root_id",
    "cave_ids",
    "cave_ids_index",
}


@dataclass(frozen=True)
class RootIdSummary:
    dtype: str
    non_null_count: int
    unique_count: int
    duplicate_rows: int
    precision_safe: bool
    normalized: tuple[int, ...] | None
    valid_indices: tuple[int, ...]


@dataclass(frozen=True)
class RaggedNucleusSummary:
    data_dtype: str
    index_dtype: str
    raw_value_count: int
    row_count: int
    rows_unmatched: int
    rows_single: int
    rows_ambiguous: int
    unsafe_nonnull_values: int
    precision_safe: bool
    row_nucleus_ids: tuple[int | None, ...] | None


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


def _exact_positive_id(raw: Any) -> tuple[int | None, bool]:
    """Return an exact positive ID, or mark a non-null representation unsafe."""
    if isinstance(raw, np.generic):
        raw = raw.item()

    if isinstance(raw, (int, np.integer)):
        value = int(raw)
        return (value, True) if value > 0 else (None, True)

    if isinstance(raw, (float, np.floating)):
        value = float(raw)
        if np.isnan(value):
            return None, True
        if not np.isfinite(value):
            return None, False
        if value <= 0:
            return None, True
        exact = value <= MAX_EXACT_FLOAT_INTEGER and value == np.floor(value)
        return (int(value), True) if exact else (None, False)

    text = _decode_scalar(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None, True
    try:
        value = int(text)
    except ValueError:
        return None, False
    return (value, True) if value > 0 else (None, True)


def decode_ragged_nucleus_ids(
    values: np.ndarray,
    index: np.ndarray,
    *,
    row_count: int,
) -> RaggedNucleusSummary:
    """Decode NWB VectorData/VectorIndex without choosing among ambiguous IDs."""
    data = np.asarray(values).reshape(-1)
    endpoints_raw = np.asarray(index).reshape(-1)
    if len(endpoints_raw) != row_count:
        raise RuntimeError("cave_ids_index length does not match PlaneSegmentation rows")
    if endpoints_raw.dtype.kind not in "iu":
        raise RuntimeError("cave_ids_index is not an exact integer dtype")

    endpoints = [int(value) for value in endpoints_raw]
    previous = 0
    for endpoint in endpoints:
        if endpoint < previous or endpoint > len(data):
            raise RuntimeError("cave_ids_index contains invalid cumulative endpoints")
        previous = endpoint
    if endpoints and endpoints[-1] != len(data):
        raise RuntimeError("cave_ids_index final endpoint does not consume cave_ids data")
    if not endpoints and len(data):
        raise RuntimeError("cave_ids contains values without VectorIndex rows")

    row_ids: list[int | None] = []
    rows_unmatched = 0
    rows_single = 0
    rows_ambiguous = 0
    unsafe_values = 0
    start = 0
    for endpoint in endpoints:
        unique_ids: set[int] = set()
        for raw in data[start:endpoint]:
            nucleus_id, safe = _exact_positive_id(raw)
            if not safe:
                unsafe_values += 1
            if nucleus_id is not None:
                unique_ids.add(nucleus_id)
        start = endpoint

        if not unique_ids:
            row_ids.append(None)
            rows_unmatched += 1
        elif len(unique_ids) == 1:
            row_ids.append(next(iter(unique_ids)))
            rows_single += 1
        else:
            row_ids.append(None)
            rows_ambiguous += 1

    precision_safe = unsafe_values == 0
    return RaggedNucleusSummary(
        data_dtype=str(data.dtype),
        index_dtype=str(endpoints_raw.dtype),
        raw_value_count=int(len(data)),
        row_count=row_count,
        rows_unmatched=rows_unmatched,
        rows_single=rows_single,
        rows_ambiguous=rows_ambiguous,
        unsafe_nonnull_values=unsafe_values,
        precision_safe=precision_safe,
        row_nucleus_ids=tuple(row_ids) if precision_safe else None,
    )


def cohort_csv_bytes(rows: list[dict[str, object]]) -> bytes:
    """Create a deterministic one-ROI/one-nucleus structural cohort manifest."""
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["plane"]),
            str(row["roi_id"]),
            int(row["nucleus_id"]),
        ),
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=[
            "asset_path",
            "plane",
            "roi_id",
            "nucleus_id",
            "coreg_source",
            "v117_pt_root_id",
        ],
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


def _deduplicate_nucleus_rows(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    counts = Counter(int(row["nucleus_id"]) for row in rows)
    duplicated_ids = {nucleus_id for nucleus_id, count in counts.items() if count > 1}
    retained = [row for row in rows if int(row["nucleus_id"]) not in duplicated_ids]
    return retained, {
        "duplicate_nucleus_identities": len(duplicated_ids),
        "roi_rows_excluded_for_duplicate_nucleus": len(rows) - len(retained),
    }


def scan_hdf5_metadata(
    h5: Any,
    *,
    asset_path: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
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
    candidate_rows: list[dict[str, object]] = []
    all_nucleus_ids_safe = True
    all_root_ids_exact = True

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

        if "pt_root_id" in plane:
            root_path = f"/processing/ophys/ImageSegmentation/{plane_name}/pt_root_id"
            root_values = _read_allowed_value(plane["pt_root_id"], root_path)
            value_read_paths.append(root_path)
            roots = summarize_root_ids(root_values)
            all_root_ids_exact = all_root_ids_exact and roots.precision_safe
            report["pt_root_id"] = {
                "present": True,
                "dtype": roots.dtype,
                "non_null_count": roots.non_null_count,
                "unique_count": roots.unique_count,
                "duplicate_rows": roots.duplicate_rows,
                "precision_safe_for_exact_integer_identity": roots.precision_safe,
                "canonical_identity": False,
            }
        else:
            report["pt_root_id"] = {"present": False, "canonical_identity": False}

        required = {"id", "cave_ids", "cave_ids_index"}
        if not required.issubset(plane.keys()):
            report["nucleus_coregistration"] = {"present": False}
            plane_reports.append(report)
            continue

        roi_path = f"/processing/ophys/ImageSegmentation/{plane_name}/id"
        cave_path = f"/processing/ophys/ImageSegmentation/{plane_name}/cave_ids"
        index_path = f"/processing/ophys/ImageSegmentation/{plane_name}/cave_ids_index"
        roi_values = _read_allowed_value(plane["id"], roi_path).reshape(-1)
        cave_values = _read_allowed_value(plane["cave_ids"], cave_path)
        cave_index = _read_allowed_value(plane["cave_ids_index"], index_path)
        value_read_paths.extend([roi_path, cave_path, index_path])

        ragged = decode_ragged_nucleus_ids(
            cave_values,
            cave_index,
            row_count=row_count,
        )
        all_nucleus_ids_safe = all_nucleus_ids_safe and ragged.precision_safe
        report["nucleus_coregistration"] = {
            "present": True,
            "source_column": "cave_ids",
            "canonical_identity": "nucleus_id",
            "data_dtype": ragged.data_dtype,
            "index_dtype": ragged.index_dtype,
            "raw_value_count": ragged.raw_value_count,
            "rows_unmatched": ragged.rows_unmatched,
            "rows_single_unique_id": ragged.rows_single,
            "rows_ambiguous_multiple_ids": ragged.rows_ambiguous,
            "unsafe_nonnull_values": ragged.unsafe_nonnull_values,
            "precision_safe": ragged.precision_safe,
        }

        if ragged.precision_safe and ragged.row_nucleus_ids is not None:
            for source_index, nucleus_id in enumerate(ragged.row_nucleus_ids):
                if nucleus_id is None:
                    continue
                candidate_rows.append(
                    {
                        "asset_path": asset_path,
                        "plane": plane_name,
                        "roi_id": _decode_scalar(roi_values[source_index]),
                        "nucleus_id": nucleus_id,
                        "coreg_source": COREG_SOURCE,
                    }
                )
        plane_reports.append(report)

    deduplicated_rows, duplicate_report = _deduplicate_nucleus_rows(candidate_rows)
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

    report = {
        "nwb_scalar_metadata": scalar_metadata,
        "plane_segmentations": plane_reports,
        "roi_response_series_metadata_only": response_series,
        "stimulus_interval_metadata_only": interval_reports,
        "coregistration": {
            "canonical_identity": "nucleus_id",
            "coreg_source": COREG_SOURCE,
            "nucleus_id_precision_safe": all_nucleus_ids_safe,
            "nwb_pt_root_id_exact_integer_safe": all_root_ids_exact,
            "pre_duplicate_candidate_rows": len(candidate_rows),
            "pre_static_candidate_rows": (
                len(deduplicated_rows) if all_nucleus_ids_safe else 0
            ),
            **duplicate_report,
        },
        "value_read_paths": value_read_paths,
        "functional_values_read": False,
    }
    return report, deduplicated_rows if all_nucleus_ids_safe else []


def _parse_exact_decimal(text: str, *, field: str) -> int:
    value = text.strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        raise RuntimeError(f"static nucleus table has missing {field}")
    if any(character in value.lower() for character in ("e", ".")):
        raise RuntimeError(f"static nucleus table {field} is not exact decimal integer text")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"static nucleus table has invalid {field}: {value!r}") from exc
    if parsed <= 0:
        raise RuntimeError(f"static nucleus table has non-positive {field}")
    return parsed


def validate_static_nucleus_table(
    raw_csv: bytes,
    candidate_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Validate stable nucleus IDs against the public v117 static nucleus table."""
    target_ids = {int(row["nucleus_id"]) for row in candidate_rows}
    target_hits: Counter[int] = Counter()
    target_roots: dict[int, int] = {}
    total_rows = 0

    text = io.StringIO(raw_csv.decode("utf-8-sig"))
    reader = csv.DictReader(text)
    if reader.fieldnames is None or not {"id", "pt_root_id"}.issubset(reader.fieldnames):
        raise RuntimeError("static nucleus table is missing id or pt_root_id")

    for source_row in reader:
        total_rows += 1
        nucleus_id = _parse_exact_decimal(source_row["id"], field="id")
        if nucleus_id not in target_ids:
            continue
        root_id = _parse_exact_decimal(source_row["pt_root_id"], field="pt_root_id")
        target_hits[nucleus_id] += 1
        target_roots[nucleus_id] = root_id

    missing = sorted(target_ids - set(target_hits))
    nonunique = sorted(nucleus_id for nucleus_id, count in target_hits.items() if count != 1)
    validation_ok = not missing and not nonunique and bool(target_ids)

    validated_rows: list[dict[str, object]] = []
    if validation_ok:
        for row in candidate_rows:
            nucleus_id = int(row["nucleus_id"])
            validated_rows.append(
                {
                    **row,
                    "v117_pt_root_id": target_roots[nucleus_id],
                }
            )

    return validated_rows, {
        "url": NUCLEUS_DETECTION_URL,
        "sha256": hashlib.sha256(raw_csv).hexdigest(),
        "source_rows": total_rows,
        "candidate_unique_nucleus_ids": len(target_ids),
        "matched_unique_nucleus_ids": len(target_hits),
        "missing_candidate_ids": len(missing),
        "nonunique_candidate_ids": len(nonunique),
        "validation_ok": validation_ok,
    }


def _download_static_nucleus_table() -> bytes:
    request = urllib.request.Request(
        NUCLEUS_DETECTION_URL,
        headers={"User-Agent": "fcr-experiment-009/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read()


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
            nwb_report, candidate_rows = scan_hdf5_metadata(
                h5,
                asset_path=FROZEN_ASSET_PATH,
            )

    static_csv = _download_static_nucleus_table()
    validated_rows, static_report = validate_static_nucleus_table(
        static_csv,
        candidate_rows,
    )
    manifest_bytes = cohort_csv_bytes(validated_rows) if static_report["validation_ok"] else None
    nwb_report["coregistration"]["static_nucleus_validation"] = static_report
    nwb_report["coregistration"]["candidate_rows"] = (
        len(validated_rows) if manifest_bytes is not None else 0
    )
    nwb_report["coregistration"]["candidate_manifest_sha256"] = (
        hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes is not None else None
    )

    report = {
        "experiment": "009",
        "stage": "A-outcome-blind-coregistration-preflight",
        "preregistration_issue": 27,
        "schema_reconciliation_comment": 5404064572,
        "duplicate_mapping_addendum_comment": 5404074984,
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
