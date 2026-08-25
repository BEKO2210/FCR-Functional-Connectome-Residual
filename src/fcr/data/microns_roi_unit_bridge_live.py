"""Live orchestration for the reconciled Experiment 010 Stage-A identity bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fcr.data.microns_asset_identity import scan_nwb_roi_identity_from_asset_path
from fcr.data.microns_roi_unit_bridge import (
    COREG_SOURCE,
    DANDISET_ID,
    DANDISET_VERSION,
    FROZEN_ASSET_PATH,
    PREREGISTRATION_ISSUE,
    RELEASE_COREG_URL,
    SCANUNIT_URL,
    _fetch_bytes,
    _json_safe,
    cohort_csv_bytes,
    enforce_one_to_one_structural_cells,
    join_units_to_release,
    load_scanunit_dataframe,
    map_rois_to_units,
    parse_release_coregistration,
)

SCHEMA_RECONCILIATION_COMMENT = 5404679857


def run_reconciled_preflight(
    output_json: str | Path,
    output_csv: str | Path,
) -> dict[str, object]:
    """Run Experiment 010 with session/scan identity frozen from the DANDI asset path."""
    try:
        import fsspec
        import h5py
        from dandi.dandiapi import DandiAPIClient
    except ImportError as exc:  # pragma: no cover - live workflow only
        raise RuntimeError(
            "Experiment 010 live preflight requires dandi, fsspec, and h5py"
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
            nwb_report, roi_rows = scan_nwb_roi_identity_from_asset_path(h5)

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
        "schema_reconciliation_comment": SCHEMA_RECONCILIATION_COMMENT,
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
