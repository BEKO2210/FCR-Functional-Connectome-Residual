"""Self-describing archived MICrONS v343 functional coregistration reader.

Experiment 010 uses the official archived data/header pair for the historical
`func_unit_em_match_release` table. The header sidecar determines physical
column order; data values are never used to infer schema.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
from collections import defaultdict

from fcr.data.microns_roi_unit_bridge import (
    EXPECTED_SCAN_IDX,
    EXPECTED_SESSION,
    _exact_integer,
)

V343_COREG_DATA_URL = (
    "https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/v343/"
    "func_unit_em_match_release_merged.csv.gz"
)
V343_COREG_HEADER_URL = (
    "https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/v343/"
    "func_unit_em_match_release_merged_header.csv"
)
V343_RECONCILIATION_COMMENT = 5408791124

EXPECTED_FIELDS = {
    "id",
    "valid",
    "pt_position_x",
    "pt_position_y",
    "pt_position_z",
    "pt_supervoxel_id",
    "pt_root_id",
    "session",
    "scan_idx",
    "unit_id",
}


def parse_v343_header(header_bytes: bytes) -> tuple[list[str], dict[str, str]]:
    """Parse the archived `(column, type)` sidecar without inferring data order."""
    try:
        text = header_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RuntimeError("v343 coregistration header is not UTF-8") from exc

    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows:
        raise RuntimeError("v343 coregistration header is empty")

    columns: list[str] = []
    types: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        if len(row) != 2:
            raise RuntimeError(
                f"v343 coregistration header row {index} does not contain exactly two fields"
            )
        column = row[0].strip()
        dtype = row[1].strip()
        if not column or not dtype:
            raise RuntimeError(f"v343 coregistration header row {index} contains an empty field")
        if column in types:
            raise RuntimeError(f"v343 coregistration header duplicates column {column!r}")
        columns.append(column)
        types[column] = dtype

    if set(columns) != EXPECTED_FIELDS:
        missing = sorted(EXPECTED_FIELDS - set(columns))
        extra = sorted(set(columns) - EXPECTED_FIELDS)
        raise RuntimeError(
            "unexpected v343 func_unit_em_match_release field set: "
            f"missing={missing}, extra={extra}"
        )
    if len(columns) != len(EXPECTED_FIELDS):
        raise RuntimeError("v343 coregistration header does not contain exactly ten columns")

    return columns, types


def parse_v343_coregistration(
    data_bytes: bytes,
    header_bytes: bytes,
) -> tuple[dict[int, set[tuple[int, int, int, int]]], dict[str, object]]:
    """Parse v343 archived coregistration using only sidecar-defined column order."""
    columns, types = parse_v343_header(header_bytes)

    try:
        raw = gzip.decompress(data_bytes)
    except (OSError, EOFError) as exc:
        raise RuntimeError("v343 coregistration data is not valid gzip") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("v343 coregistration data is not UTF-8 after gzip decode") from exc

    by_unit: dict[int, set[tuple[int, int, int, int]]] = defaultdict(set)
    source_rows = 0
    target_rows = 0
    reader = csv.reader(io.StringIO(text, newline=""))
    for row_number, values in enumerate(reader, start=1):
        if len(values) != len(columns):
            raise RuntimeError(
                "v343 coregistration data row width mismatch at row "
                f"{row_number}: expected {len(columns)}, got {len(values)}"
            )
        record = dict(zip(columns, values, strict=True))
        source_rows += 1

        session = _exact_integer(record["session"], minimum=0)
        scan_idx = _exact_integer(record["scan_idx"], minimum=0)
        if session != EXPECTED_SESSION or scan_idx != EXPECTED_SCAN_IDX:
            continue

        target_rows += 1
        if record["valid"].strip().lower() not in {"t", "true", "1"}:
            raise RuntimeError(
                f"target v343 functional coregistration row {row_number} is marked invalid"
            )

        unit_id = _exact_integer(record["unit_id"], minimum=0)
        root_id = _exact_integer(record["pt_root_id"], minimum=1)
        x = _exact_integer(record["pt_position_x"], minimum=0)
        y = _exact_integer(record["pt_position_y"], minimum=0)
        z = _exact_integer(record["pt_position_z"], minimum=0)
        by_unit[unit_id].add((root_id, x, y, z))

    report: dict[str, object] = {
        "archive_materialization_version": 343,
        "table": "func_unit_em_match_release",
        "data_url": V343_COREG_DATA_URL,
        "header_url": V343_COREG_HEADER_URL,
        "data_size_bytes": len(data_bytes),
        "header_size_bytes": len(header_bytes),
        "data_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
        "schema": columns,
        "schema_types": types,
        "source_rows": source_rows,
        "target_session_scan_rows": target_rows,
        "reconciliation_comment": V343_RECONCILIATION_COMMENT,
    }
    return by_unit, report
