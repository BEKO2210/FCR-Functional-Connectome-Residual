"""Outcome-blind NWB asset identity reconciliation for Experiment 010."""

from __future__ import annotations

import re
from typing import Any

from fcr.data.microns_roi_unit_bridge import (
    EXPECTED_SCAN_IDX,
    EXPECTED_SESSION,
    FROZEN_ASSET_PATH,
    _exact_integer,
    _parse_session_id,
    _read_allowed_nwb_value,
)

_ASSET_SESSION_SCAN = re.compile(r"_ses-([0-9]+)-scan-([0-9]+)_")
_PLANE_NAME = re.compile(r"^PlaneSegmentation([1-9][0-9]*)$")


def parse_frozen_asset_session_scan(
    asset_path: str = FROZEN_ASSET_PATH,
) -> tuple[int, int]:
    """Parse the preregistered session/scan pair from the frozen DANDI asset path."""
    filename = asset_path.rsplit("/", 1)[-1]
    matches = list(_ASSET_SESSION_SCAN.finditer(filename))
    if len(matches) != 1:
        raise RuntimeError(
            "frozen DANDI asset path does not contain exactly one session/scan identity"
        )
    session = int(matches[0].group(1))
    scan_idx = int(matches[0].group(2))
    if session != EXPECTED_SESSION or scan_idx != EXPECTED_SCAN_IDX:
        raise RuntimeError(
            "asset-path identity contradicts Experiment 010 preregistration: "
            f"session={session}, scan_idx={scan_idx}"
        )
    return session, scan_idx


def scan_nwb_roi_identity_from_asset_path(
    h5: Any,
    *,
    asset_path: str = FROZEN_ASSET_PATH,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Read only ROI IDs, using the frozen asset path for canonical session/scan identity."""
    session, scan_idx = parse_frozen_asset_session_scan(asset_path)
    value_read_paths: list[str] = []
    session_id_present = "session_id" in h5
    session_id_text: str | None = None

    if session_id_present:
        raw = _read_allowed_nwb_value(h5["session_id"], "/session_id")
        checked_session, checked_scan, session_id_text = _parse_session_id(raw)
        value_read_paths.append("/session_id")
        if checked_session != session or checked_scan != scan_idx:
            raise RuntimeError("NWB session_id disagrees with frozen asset-path identity")

    try:
        segmentation = h5["processing"]["ophys"]["ImageSegmentation"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("frozen NWB ImageSegmentation hierarchy is missing") from exc

    rows: list[dict[str, object]] = []
    planes: list[dict[str, object]] = []
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
        "session": session,
        "scan_idx": scan_idx,
        "session_identity_source": "frozen-dandi-asset-path",
        "session_id_present_in_nwb": session_id_present,
        "session_id_diagnostic": session_id_text,
        "asset_path_identity_pattern": "_ses-([0-9]+)-scan-([0-9]+)_",
        "plane_segmentations": planes,
        "total_roi_rows": len(rows),
        "value_read_paths": value_read_paths,
        "functional_values_read": False,
    }
    return report, rows
