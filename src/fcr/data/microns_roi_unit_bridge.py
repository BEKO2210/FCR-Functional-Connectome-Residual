"""Outcome-blind provenance-native MICrONS identity bridge for Experiment 010.

Stage A may read NWB scalar identity metadata and PlaneSegmentation row IDs, then
join them through the public ScanUnit mapping and the frozen v117 functional
co-registration release. It must never read functional response or connectivity values.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DANDISET_ID = "000402"
DANDISET_VERSION = "draft"
FROZEN_ASSET_PATH = "sub-17797/sub-17797_ses-9-scan-4_behavior+image+ophys.nwb"
EXPECTED_SESSION = 9
EXPECTED_SCAN_IDX = 4
PREREGISTRATION_ISSUE = 29

SCANUNIT_COMMIT = "86268039695669a216b0a1959c9812b5e94c2eb5"
SCANUNIT_GIT_BLOB_SHA1 = "0e960c44695f79881296322a1bdb972dcae755ee"
SCANUNIT_SIZE_BYTES = 14_870_416
SCANUNIT_URL = (
    "https://raw.githubusercontent.com/dandi/example-notebooks/"
    f"{SCANUNIT_COMMIT}/000402/MICrONS/coregistration/ScanUnit.pkl"
)
SCANUNIT_REQUIRED_COLUMNS = ("session", "scan_idx", "unit_id", "field", "mask_id")

RELEASE_COREG_URL = (
    "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/"
    "functional_coregistration/func_unit_em_match_release.csv"
)
RELEASE_COREG_COLUMNS = (
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
)

COREG_SOURCE = "release-func-unit-em-match-v117"
MAX_EXACT_FLOAT_INTEGER = 2**53
_PLANE_NAME = re.compile(r"^PlaneSegmentation([1-9][0-9]*)$")

_MANIFEST_FIELDS = (
    "asset_path",
    "plane",
    "roi_id",
    "mask_id",
    "session",
    "scan_idx",
    "field",
    "unit_id",
    "v117_pt_root_id",
    "pt_position_x",
    "pt_position_y",
    "pt_position_z",
    "coreg_source",
)


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_scalar(value.item())
    if isinstance(value, np.generic):
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


def _exact_integer(raw: Any, *, minimum: int | None = None) -> int:
    """Decode an integer without accepting lossy float or decimal-string identities."""
    if isinstance(raw, np.generic):
        raw = raw.item()

    if isinstance(raw, bool):
        raise RuntimeError("boolean value cannot be used as an exact identity")

    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, float):
        if not np.isfinite(raw) or abs(raw) > MAX_EXACT_FLOAT_INTEGER or raw != np.floor(raw):
            raise RuntimeError(f"unsafe non-integral identity value: {raw!r}")
        value = int(raw)
    else:
        text = _decode_scalar(raw).strip()
        if not re.fullmatch(r"-?(0|[1-9][0-9]*)", text):
            raise RuntimeError(f"non-decimal exact identity value: {text!r}")
        value = int(text)

    if minimum is not None and value < minimum:
        raise RuntimeError(f"exact identity value {value} is below minimum {minimum}")
    return value


def _read_allowed_nwb_value(dataset: Any, path: str) -> np.ndarray:
    """Read only Experiment-010-whitelisted NWB identity values."""
    normalized = "/" + path.strip("/")
    plane_id = (
        normalized.startswith("/processing/ophys/ImageSegmentation/PlaneSegmentation")
        and normalized.endswith("/id")
    )
    if normalized != "/session_id" and not plane_id:
        raise RuntimeError(f"Experiment 010 Stage A forbids NWB value read from {normalized}")
    return np.asarray(dataset[()])


def _parse_session_id(value: Any) -> tuple[int, int, str]:
    text = _decode_scalar(value).strip()
    parts = text.split("-")
    if len(parts) < 3:
        raise RuntimeError(f"unexpected NWB session_id format: {text!r}")
    session = _exact_integer(parts[0], minimum=0)
    scan_idx = _exact_integer(parts[2], minimum=0)
    if session != EXPECTED_SESSION or scan_idx != EXPECTED_SCAN_IDX:
        raise RuntimeError(
            "NWB session_id contradicts frozen Experiment 010 asset: "
            f"parsed session={session}, scan_idx={scan_idx}"
        )
    return session, scan_idx, text


def scan_nwb_roi_identity(
    h5: Any,
    *,
    asset_path: str = FROZEN_ASSET_PATH,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Read only session identity and PlaneSegmentation IDs from an NWB HDF5 handle."""
    if "session_id" not in h5:
        raise RuntimeError("frozen NWB has no session_id scalar")
    session_raw = _read_allowed_nwb_value(h5["session_id"], "/session_id")
    session, scan_idx, session_text = _parse_session_id(session_raw)

    try:
        segmentation = h5["processing"]["ophys"]["ImageSegmentation"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("frozen NWB ImageSegmentation hierarchy is missing") from exc

    rows: list[dict[str, object]] = []
    planes: list[dict[str, object]] = []
    value_read_paths = ["/session_id"]

    for plane_name in sorted(segmentation.keys()):
        plane = segmentation[plane_name]
        match = _PLANE_NAME.fullmatch(str(plane_name))
        if match is None:
            raise RuntimeError(f"non-conforming PlaneSegmentation name: {plane_name!r}")
        field = int(match.group(1))
        if "id" not in plane:
            raise RuntimeError(f"{plane_name} has no DynamicTable id column")

        path = f"/processing/ophys/ImageSegmentation/{plane_name}/id"
        ids = _read_allowed_nwb_value(plane["id"], path).reshape(-1)
        value_read_paths.append(path)
        mask_ids = [_exact_integer(raw, minimum=0) for raw in ids]
        if len(mask_ids) != len(set(mask_ids)):
            raise RuntimeError(f"{plane_name} contains duplicate DynamicTable ids")

        for mask_id in mask_ids:
            rows.append(
                {
                    "asset_path": asset_path,
                    "plane": str(plane_name),
                    "roi_id": mask_id,
                    "mask_id": mask_id,
                    "session": session,
                    "scan_idx": scan_idx,
                    "field": field,
                }
            )
        planes.append(
            {
                "name": str(plane_name),
                "field": field,
                "roi_rows": len(mask_ids),
                "id_dtype": str(ids.dtype),
            }
        )

    if not rows:
        raise RuntimeError("frozen NWB contains no PlaneSegmentation ROI rows")

    report: dict[str, object] = {
        "session_id": session_text,
        "session": session,
        "scan_idx": scan_idx,
        "plane_segmentations": planes,
        "total_roi_rows": len(rows),
        "value_read_paths": value_read_paths,
        "functional_values_read": False,
    }
    return report, rows


def _git_blob_sha1(data: bytes) -> str:
    prefix = f"blob {len(data)}\0".encode()
    return hashlib.sha1(prefix + data).hexdigest()


def verify_scanunit_bytes(data: bytes) -> dict[str, object]:
    """Verify immutable Git provenance before the pickle is deserialized."""
    if len(data) != SCANUNIT_SIZE_BYTES:
        raise RuntimeError(
            f"ScanUnit byte size changed: expected {SCANUNIT_SIZE_BYTES}, got {len(data)}"
        )
    blob_sha = _git_blob_sha1(data)
    if blob_sha != SCANUNIT_GIT_BLOB_SHA1:
        raise RuntimeError(
            f"ScanUnit Git blob changed: expected {SCANUNIT_GIT_BLOB_SHA1}, got {blob_sha}"
        )
    return {
        "url": SCANUNIT_URL,
        "commit": SCANUNIT_COMMIT,
        "git_blob_sha1": blob_sha,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def load_scanunit_dataframe(data: bytes) -> tuple[pd.DataFrame, dict[str, object]]:
    """Verify the pinned official pickle, then deserialize it."""
    provenance = verify_scanunit_bytes(data)
    frame = pd.read_pickle(io.BytesIO(data))
    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError("ScanUnit.pkl did not deserialize to a pandas DataFrame")
    missing = [name for name in SCANUNIT_REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"ScanUnit schema missing required columns: {missing}")
    provenance["columns"] = [str(name) for name in frame.columns]
    provenance["source_rows"] = int(len(frame))
    return frame, provenance


def map_rois_to_units(
    roi_rows: list[dict[str, object]],
    scanunit: pd.DataFrame,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Exact-join frozen NWB ROI keys to functional unit IDs."""
    session_values = pd.to_numeric(scanunit["session"], errors="coerce")
    scan_values = pd.to_numeric(scanunit["scan_idx"], errors="coerce")
    target = scanunit[(session_values == EXPECTED_SESSION) & (scan_values == EXPECTED_SCAN_IDX)]

    mapping: dict[tuple[int, int, int, int], set[int]] = defaultdict(set)
    for record in target.to_dict(orient="records"):
        session = _exact_integer(record["session"], minimum=0)
        scan_idx = _exact_integer(record["scan_idx"], minimum=0)
        field = _exact_integer(record["field"], minimum=1)
        mask_id = _exact_integer(record["mask_id"], minimum=0)
        unit_id = _exact_integer(record["unit_id"], minimum=0)
        mapping[(session, scan_idx, field, mask_id)].add(unit_id)

    conflicting = {key: ids for key, ids in mapping.items() if len(ids) > 1}
    if conflicting:
        raise RuntimeError(
            f"ScanUnit contains {len(conflicting)} conflicting target identity keys"
        )

    mapped: list[dict[str, object]] = []
    unmapped = 0
    for row in roi_rows:
        key = (
            int(row["session"]),
            int(row["scan_idx"]),
            int(row["field"]),
            int(row["mask_id"]),
        )
        units = mapping.get(key)
        if not units:
            unmapped += 1
            continue
        unit_id = next(iter(units))
        mapped.append({**row, "unit_id": unit_id})

    report = {
        "target_source_rows": int(len(target)),
        "exact_match_roi_rows": len(mapped),
        "unmapped_roi_rows": unmapped,
        "conflicting_target_keys": 0,
    }
    return mapped, report


def parse_release_coregistration(
    data: bytes,
) -> tuple[dict[int, set[tuple[int, int, int, int]]], dict[str, object]]:
    """Parse the frozen public v117 functional-unit to EM release exactly."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("v117 functional coregistration release is not UTF-8") from exc

    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = tuple(reader.fieldnames or ())
    if fieldnames != RELEASE_COREG_COLUMNS:
        raise RuntimeError(
            "unexpected func_unit_em_match_release schema: "
            f"expected {RELEASE_COREG_COLUMNS}, got {fieldnames}"
        )

    by_unit: dict[int, set[tuple[int, int, int, int]]] = defaultdict(set)
    source_rows = 0
    target_rows = 0
    for record in reader:
        source_rows += 1
        session = _exact_integer(record["session"], minimum=0)
        scan_idx = _exact_integer(record["scan_idx"], minimum=0)
        if session != EXPECTED_SESSION or scan_idx != EXPECTED_SCAN_IDX:
            continue
        target_rows += 1
        if str(record["valid"]).strip().lower() not in {"t", "true", "1"}:
            raise RuntimeError("target v117 functional coregistration row is marked invalid")
        unit_id = _exact_integer(record["unit_id"], minimum=0)
        root_id = _exact_integer(record["pt_root_id"], minimum=1)
        x = _exact_integer(record["pt_position_x"], minimum=0)
        y = _exact_integer(record["pt_position_y"], minimum=0)
        z = _exact_integer(record["pt_position_z"], minimum=0)
        by_unit[unit_id].add((root_id, x, y, z))

    report: dict[str, object] = {
        "url": RELEASE_COREG_URL,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "schema": list(fieldnames),
        "source_rows": source_rows,
        "target_session_scan_rows": target_rows,
    }
    return by_unit, report


def join_units_to_release(
    unit_rows: list[dict[str, object]],
    by_unit: dict[int, set[tuple[int, int, int, int]]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Apply the frozen release membership and one-structural-match rule."""
    accepted: list[dict[str, object]] = []
    unmatched = 0
    ambiguous_roi_rows = 0
    ambiguous_units: set[int] = set()

    for row in unit_rows:
        unit_id = int(row["unit_id"])
        matches = by_unit.get(unit_id, set())
        if not matches:
            unmatched += 1
            continue
        if len(matches) != 1:
            ambiguous_units.add(unit_id)
            ambiguous_roi_rows += 1
            continue
        root_id, x, y, z = next(iter(matches))
        accepted.append(
            {
                **row,
                "v117_pt_root_id": root_id,
                "pt_position_x": x,
                "pt_position_y": y,
                "pt_position_z": z,
                "coreg_source": COREG_SOURCE,
            }
        )

    report = {
        "exact_release_match_roi_rows": len(accepted),
        "unmatched_release_roi_rows": unmatched,
        "ambiguous_release_units": len(ambiguous_units),
        "ambiguous_release_roi_rows": ambiguous_roi_rows,
    }
    return accepted, report


def enforce_one_to_one_structural_cells(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Exclude every ROI belonging to a structural root represented more than once."""
    counts = Counter(int(row["v117_pt_root_id"]) for row in rows)
    duplicated = {root for root, count in counts.items() if count > 1}
    kept = [row for row in rows if int(row["v117_pt_root_id"]) not in duplicated]
    excluded = len(rows) - len(kept)
    return kept, {
        "duplicate_structural_roots": len(duplicated),
        "roi_rows_excluded_for_duplicate_root": excluded,
    }


def cohort_csv_bytes(rows: list[dict[str, object]]) -> bytes:
    """Serialize the Experiment 010 primary cohort deterministically."""
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["field"]),
            int(row["mask_id"]),
            int(row["unit_id"]),
            int(row["v117_pt_root_id"]),
        ),
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_MANIFEST_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({name: row[name] for name in _MANIFEST_FIELDS} for row in ordered)
    return stream.getvalue().encode("utf-8")


def _fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "fcr-exp010/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        return response.read()


def run_preflight(
    output_json: str | Path,
    output_csv: str | Path,
) -> dict[str, object]:
    """Run the frozen Experiment 010 token-free identity preflight."""
    try:
        import fsspec
        import h5py
        from dandi.dandiapi import DandiAPIClient
    except ImportError as exc:  # pragma: no cover - live workflow only
        raise RuntimeError("Experiment 010 live preflight requires dandi, fsspec, and h5py") from exc

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
            nwb_report, roi_rows = scan_nwb_roi_identity(h5)

    scanunit_bytes = _fetch_bytes(SCANUNIT_URL)
    scanunit, scanunit_report = load_scanunit_dataframe(scanunit_bytes)
    unit_rows, scanunit_join = map_rois_to_units(roi_rows, scanunit)
    scanunit_report.update(scanunit_join)

    release_bytes = _fetch_bytes(RELEASE_COREG_URL)
    release_by_unit, release_report = parse_release_coregistration(release_bytes)
    structural_rows, release_join = join_units_to_release(unit_rows, release_by_unit)
    release_report.update(release_join)

    cohort_rows, duplicate_report = enforce_one_to_one_structural_cells(structural_rows)
    manifest = cohort_csv_bytes(cohort_rows)
    manifest_sha = hashlib.sha256(manifest).hexdigest()

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_bytes(manifest)

    report: dict[str, object] = {
        "experiment": "010",
        "stage": "A-outcome-blind-provenance-native-identity",
        "evidence_level": "E0-plumbing-only",
        "preregistration_issue": PREREGISTRATION_ISSUE,
        "functional_values_read": False,
        "connectivity_accessed": False,
        "asset": asset_report,
        "nwb": nwb_report,
        "scanunit": scanunit_report,
        "release_coregistration": release_report,
        "cohort": {
            **duplicate_report,
            "rows": len(cohort_rows),
            "manifest_sha256": manifest_sha,
            "coreg_source": COREG_SOURCE,
        },
    }

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
