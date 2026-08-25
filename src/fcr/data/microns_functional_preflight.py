"""Outcome-blind DANDI/NWB preflight for Experiment 009.

This module may read structural coregistration identifiers, exact structural point
coordinates, and scalar NWB metadata, but it must never read calcium, movie, or
stimulus-response array values.
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
STATIC_NUCLEUS_COLUMNS = (
    "id",
    "valid",
    "pt_supervoxel_id",
    "pt_root_id",
    "pt_position_x",
    "pt_position_y",
    "pt_position_z",
    "volume",
)
MAX_EXACT_FLOAT_INTEGER = 2**53
COREG_SOURCE = "exact-v117-pt-position"
LEGACY_CAVE_SOURCE = "manual_match_cave_nuclei_id"
POSITION_RECONCILIATION_COMMENT = 5404210234

_ALLOWED_VALUE_PATHS = {
    "/identifier",
    "/session_description",
    "/session_start_time",
}
_ALLOWED_PLANE_VALUE_COLUMNS = {
    "id",
    "pt_root_id",
    "pt_supervoxel_id",
    "cave_ids",
    "cave_ids_index",
    "pt_x_position",
    "pt_y_position",
    "pt_z_position",
}
_POSITION_COLUMNS = ("pt_x_position", "pt_y_position", "pt_z_position")


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
    row_is_ambiguous: tuple[bool, ...] | None


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

    if isinstance(raw, int):
        return (raw, True) if raw > 0 else (None, True)

    if isinstance(raw, float):
        if np.isnan(raw):
            return None, True
        if not np.isfinite(raw):
            return None, False
        if raw <= 0:
            return None, True
        exact = raw <= MAX_EXACT_FLOAT_INTEGER and raw == np.floor(raw)
        return (int(raw), True) if exact else (None, False)

    text = _decode_scalar(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None, True
    try:
        value = int(text)
    except ValueError:
        return None, False
    return (value, True) if value > 0 else (None, True)


def _exact_nonnegative_coordinate(raw: Any) -> tuple[int | None, bool]:
    """Decode one EM-voxel coordinate exactly; missing values are allowed."""
    if isinstance(raw, np.generic):
        raw = raw.item()

    if isinstance(raw, int):
        return (raw, True) if raw >= 0 else (None, False)

    if isinstance(raw, float):
        if np.isnan(raw):
            return None, True
        if not np.isfinite(raw) or raw < 0:
            return None, False
        exact = raw <= MAX_EXACT_FLOAT_INTEGER and raw == np.floor(raw)
        return (int(raw), True) if exact else (None, False)

    text = _decode_scalar(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None, True
    if any(character in text.lower() for character in ("e", ".")):
        return None, False
    try:
        value = int(text)
    except ValueError:
        return None, False
    return (value, True) if value >= 0 else (None, False)


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
    row_is_ambiguous: list[bool] = []
    rows_unmatched = 0
    rows_single = 0
    rows_ambiguous = 0
    unsafe_values = 0
    start = 0
    for endpoint in endpoints:
        unique_ids: set[int] = set()
        for raw in data[start:endpoint]:
            legacy_id, safe = _exact_positive_id(raw)
            if not safe:
                unsafe_values += 1
            if legacy_id is not None:
                unique_ids.add(legacy_id)
        start = endpoint

        if not unique_ids:
            row_ids.append(None)
            row_is_ambiguous.append(False)
            rows_unmatched += 1
        elif len(unique_ids) == 1:
            row_ids.append(next(iter(unique_ids)))
            row_is_ambiguous.append(False)
            rows_single += 1
        else:
            row_ids.append(None)
            row_is_ambiguous.append(True)
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
        row_is_ambiguous=tuple(row_is_ambiguous) if precision_safe else None,
    )


def cohort_csv_bytes(rows: list[dict[str, object]]) -> bytes:
    """Create the deterministic frozen Stage-A structure↔ROI cohort manifest."""
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
            "legacy_cave_id",
            "pt_position_x",
            "pt_position_y",
            "pt_position_z",
            "nucleus_id",
            "v117_pt_root_id",
            "coreg_source",
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


def _position_candidates_for_plane(
    plane: Any,
    *,
    plane_name: str,
    row_count: int,
    asset_path: str,
    ragged: RaggedNucleusSummary,
    value_read_paths: list[str],
) -> tuple[list[dict[str, object]], dict[str, int | bool]]:
    required_positions = set(_POSITION_COLUMNS)
    if not required_positions.issubset(plane.keys()):
        if ragged.rows_single or ragged.rows_ambiguous:
            raise RuntimeError(
                "legacy structural matches exist without complete v117 position columns"
            )
        return [], {
            "position_columns_present": False,
            "rows_all_position_missing": row_count,
            "rows_exact_complete_position": 0,
            "rows_partial_position": 0,
            "unsafe_position_values": 0,
            "rows_excluded_ambiguous_legacy_cave_id": 0,
        }

    roi_path = f"/processing/ophys/ImageSegmentation/{plane_name}/id"
    roi_values = _read_allowed_value(plane["id"], roi_path).reshape(-1)
    if len(roi_values) != row_count:
        raise RuntimeError("PlaneSegmentation id length does not match row count")
    if roi_path not in value_read_paths:
        value_read_paths.append(roi_path)

    coordinate_arrays: dict[str, np.ndarray] = {}
    for column in _POSITION_COLUMNS:
        path = f"/processing/ophys/ImageSegmentation/{plane_name}/{column}"
        values = _read_allowed_value(plane[column], path).reshape(-1)
        if len(values) != row_count:
            raise RuntimeError(f"{column} length does not match PlaneSegmentation rows")
        coordinate_arrays[column] = values
        value_read_paths.append(path)

    if ragged.row_nucleus_ids is None or ragged.row_is_ambiguous is None:
        raise RuntimeError("legacy cave_ids are not precision-safe")

    candidates: list[dict[str, object]] = []
    rows_missing = 0
    rows_complete = 0
    rows_partial = 0
    unsafe_values = 0
    rows_ambiguous = 0

    for row_index in range(row_count):
        decoded: list[int | None] = []
        for column in _POSITION_COLUMNS:
            value, safe = _exact_nonnegative_coordinate(coordinate_arrays[column][row_index])
            if not safe:
                unsafe_values += 1
            decoded.append(value)

        if unsafe_values:
            raise RuntimeError("structural point coordinate is not exact non-negative integer")

        present = sum(value is not None for value in decoded)
        if present == 0:
            rows_missing += 1
            continue
        if present != 3:
            rows_partial += 1
            raise RuntimeError("structural candidate has only a partial v117 point coordinate")
        if ragged.row_is_ambiguous[row_index]:
            rows_ambiguous += 1
            continue

        rows_complete += 1
        x, y, z = (int(value) for value in decoded if value is not None)
        candidates.append(
            {
                "asset_path": asset_path,
                "plane": plane_name,
                "roi_id": _decode_scalar(roi_values[row_index]),
                "legacy_cave_id": ragged.row_nucleus_ids[row_index],
                "pt_position_x": x,
                "pt_position_y": y,
                "pt_position_z": z,
                "coreg_source": COREG_SOURCE,
            }
        )

    return candidates, {
        "position_columns_present": True,
        "rows_all_position_missing": rows_missing,
        "rows_exact_complete_position": rows_complete,
        "rows_partial_position": rows_partial,
        "unsafe_position_values": unsafe_values,
        "rows_excluded_ambiguous_legacy_cave_id": rows_ambiguous,
    }


def scan_hdf5_metadata(
    h5: Any,
    *,
    asset_path: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Inspect NWB structural metadata/coregistration without reading function."""
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
    all_legacy_ids_safe = True
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
            report["legacy_cave_coregistration"] = {"present": False}
            report["position_coregistration"] = {"position_columns_present": False}
            plane_reports.append(report)
            continue

        cave_path = f"/processing/ophys/ImageSegmentation/{plane_name}/cave_ids"
        index_path = f"/processing/ophys/ImageSegmentation/{plane_name}/cave_ids_index"
        cave_values = _read_allowed_value(plane["cave_ids"], cave_path)
        cave_index = _read_allowed_value(plane["cave_ids_index"], index_path)
        value_read_paths.extend([cave_path, index_path])

        ragged = decode_ragged_nucleus_ids(cave_values, cave_index, row_count=row_count)
        all_legacy_ids_safe = all_legacy_ids_safe and ragged.precision_safe
        report["legacy_cave_coregistration"] = {
            "present": True,
            "source_column": "cave_ids",
            "canonical_identity": False,
            "data_dtype": ragged.data_dtype,
            "index_dtype": ragged.index_dtype,
            "raw_value_count": ragged.raw_value_count,
            "rows_unmatched": ragged.rows_unmatched,
            "rows_single_unique_id": ragged.rows_single,
            "rows_ambiguous_multiple_ids": ragged.rows_ambiguous,
            "unsafe_nonnull_values": ragged.unsafe_nonnull_values,
            "precision_safe": ragged.precision_safe,
        }

        if not ragged.precision_safe:
            raise RuntimeError("legacy cave_ids contain unsafe non-null values")

        plane_candidates, position_report = _position_candidates_for_plane(
            plane,
            plane_name=plane_name,
            row_count=row_count,
            asset_path=asset_path,
            ragged=ragged,
            value_read_paths=value_read_paths,
        )
        report["position_coregistration"] = position_report
        candidate_rows.extend(plane_candidates)
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
            interval_reports.append({"name": name, "row_count": row_count, "columns": columns})

    report = {
        "nwb_scalar_metadata": scalar_metadata,
        "plane_segmentations": plane_reports,
        "roi_response_series_metadata_only": response_series,
        "stimulus_interval_metadata_only": interval_reports,
        "coregistration": {
            "canonical_identity": "nucleus_detection_v0.id",
            "coreg_source": COREG_SOURCE,
            "legacy_cave_id_source": LEGACY_CAVE_SOURCE,
            "legacy_cave_id_precision_safe": all_legacy_ids_safe,
            "nwb_pt_root_id_exact_integer_safe": all_root_ids_exact,
            "pre_static_candidate_rows": len(candidate_rows),
        },
        "value_read_paths": value_read_paths,
        "functional_values_read": False,
    }
    return report, candidate_rows


def _parse_exact_decimal(
    text: str,
    *,
    field: str,
    allow_zero: bool = False,
) -> int:
    value = text.strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        raise RuntimeError(f"static nucleus table has missing {field}")
    if any(character in value.lower() for character in ("e", ".")):
        raise RuntimeError(f"static nucleus table {field} is not exact decimal integer text")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"static nucleus table has invalid {field}: {value!r}") from exc
    if parsed < 0 or (parsed == 0 and not allow_zero):
        raise RuntimeError(f"static nucleus table has invalid non-positive {field}")
    return parsed


def _candidate_point(row: dict[str, object]) -> tuple[int, int, int]:
    return (
        int(row["pt_position_x"]),
        int(row["pt_position_y"]),
        int(row["pt_position_z"]),
    )


def validate_static_nucleus_table(
    raw_csv: bytes,
    candidate_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Join ROI candidates to the public v117 table by exact EM-voxel point."""
    target_points = {_candidate_point(row) for row in candidate_rows}
    target_hits: Counter[tuple[int, int, int]] = Counter()
    target_rows: dict[tuple[int, int, int], tuple[int, int]] = {}
    total_rows = 0

    text = io.StringIO(raw_csv.decode("utf-8-sig"))
    reader = csv.reader(text)
    for raw_row in reader:
        if not raw_row:
            continue
        total_rows += 1
        if len(raw_row) != len(STATIC_NUCLEUS_COLUMNS):
            raise RuntimeError(
                "static nucleus table row does not match frozen eight-column schema"
            )
        source_row = dict(zip(STATIC_NUCLEUS_COLUMNS, raw_row, strict=True))
        point = (
            _parse_exact_decimal(
                source_row["pt_position_x"], field="pt_position_x", allow_zero=True
            ),
            _parse_exact_decimal(
                source_row["pt_position_y"], field="pt_position_y", allow_zero=True
            ),
            _parse_exact_decimal(
                source_row["pt_position_z"], field="pt_position_z", allow_zero=True
            ),
        )
        if point not in target_points:
            continue
        nucleus_id = _parse_exact_decimal(source_row["id"], field="id")
        root_id = _parse_exact_decimal(
            source_row["pt_root_id"], field="pt_root_id", allow_zero=True
        )
        target_hits[point] += 1
        target_rows[point] = (nucleus_id, root_id)

    missing = sorted(target_points - set(target_hits))
    nonunique = sorted(point for point, count in target_hits.items() if count != 1)
    exact_join_ok = not missing and not nonunique and bool(target_points)

    joined_rows: list[dict[str, object]] = []
    no_positive_root_points: set[tuple[int, int, int]] = set()
    if exact_join_ok:
        for row in candidate_rows:
            point = _candidate_point(row)
            nucleus_id, root_id = target_rows[point]
            if root_id <= 0:
                no_positive_root_points.add(point)
                continue
            joined_rows.append(
                {
                    **row,
                    "nucleus_id": nucleus_id,
                    "v117_pt_root_id": root_id,
                }
            )

    nucleus_counts = Counter(int(row["nucleus_id"]) for row in joined_rows)
    duplicated_nucleus_ids = {
        nucleus_id for nucleus_id, count in nucleus_counts.items() if count > 1
    }
    validated_rows = [
        row for row in joined_rows if int(row["nucleus_id"]) not in duplicated_nucleus_ids
    ]

    validation_ok = exact_join_ok and bool(validated_rows)
    return validated_rows, {
        "url": NUCLEUS_DETECTION_URL,
        "file_format": "headerless-positional-v117-eight-columns",
        "column_order": list(STATIC_NUCLEUS_COLUMNS),
        "join_key": ["pt_position_x", "pt_position_y", "pt_position_z"],
        "coordinate_space": "v117-em-voxels-x4nm-y4nm-z40nm",
        "sha256": hashlib.sha256(raw_csv).hexdigest(),
        "source_rows": total_rows,
        "candidate_unique_points": len(target_points),
        "matched_unique_points": len(target_hits),
        "missing_candidate_points": len(missing),
        "nonunique_candidate_points": len(nonunique),
        "candidate_points_without_positive_v117_root": len(no_positive_root_points),
        "roi_rows_excluded_without_positive_v117_root": sum(
            1 for row in candidate_rows if _candidate_point(row) in no_positive_root_points
        ),
        "duplicate_nucleus_identities": len(duplicated_nucleus_ids),
        "roi_rows_excluded_for_duplicate_nucleus": len(joined_rows) - len(validated_rows),
        "validated_cohort_rows": len(validated_rows),
        "exact_position_join_ok": exact_join_ok,
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
    validated_rows, static_report = validate_static_nucleus_table(static_csv, candidate_rows)
    manifest_bytes = cohort_csv_bytes(validated_rows) if static_report["validation_ok"] else None
    coreg = nwb_report["coregistration"]
    coreg["static_nucleus_validation"] = static_report
    coreg["candidate_rows"] = len(validated_rows) if manifest_bytes is not None else 0
    coreg["candidate_manifest_sha256"] = (
        hashlib.sha256(manifest_bytes).hexdigest() if manifest_bytes is not None else None
    )
    coreg["duplicate_nucleus_identities"] = static_report[
        "duplicate_nucleus_identities"
    ]
    coreg["roi_rows_excluded_for_duplicate_nucleus"] = static_report[
        "roi_rows_excluded_for_duplicate_nucleus"
    ]

    report = {
        "experiment": "009",
        "stage": "A-outcome-blind-coregistration-preflight",
        "preregistration_issue": 27,
        "schema_reconciliation_comment": 5404064572,
        "duplicate_mapping_addendum_comment": 5404074984,
        "static_schema_reconciliation_comment": 5404134131,
        "position_join_reconciliation_comment": POSITION_RECONCILIATION_COMMENT,
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
