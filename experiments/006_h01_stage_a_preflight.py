"""Experiment 006 Stage A: H01 schema/storage preflight with no edge-content access."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from fcr.data.h01_preflight import (
    choose_connectivity_source,
    inspect_soma_csv,
    summarize_object_metadata,
)


H01_BUCKET = "h01-release"
SYNAPSE_PREFIX = "data/20210601/c3/synapses/exported/json/"
CREST_URL = (
    "https://storage.googleapis.com/h01_paper_public_files/"
    "CREST_browsing_database_goog14r0s5c3_eirepredict2023.db"
)
CREST_LIMIT_BYTES = 2 * 1024**3


def _list_public_gcs_objects(bucket: str, prefix: str) -> list[dict[str, Any]]:
    """List only GCS object metadata (name,size) from a public bucket."""
    endpoint = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    token: str | None = None
    items: list[dict[str, Any]] = []
    while True:
        params = {
            "prefix": prefix,
            "fields": "items(name,size),nextPageToken",
            "maxResults": "1000",
        }
        if token:
            params["pageToken"] = token
        url = endpoint + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed Google endpoint
            payload = json.load(response)
        items.extend(payload.get("items", []))
        token = payload.get("nextPageToken")
        if not token:
            return items


def _head_metadata(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed URL
            headers = response.headers
            length = headers.get("Content-Length")
            return {
                "available": True,
                "status": int(getattr(response, "status", 200)),
                "content_length": int(length) if length is not None else None,
                "content_type": headers.get("Content-Type"),
                "etag": headers.get("ETag"),
                "last_modified": headers.get("Last-Modified"),
                "accept_ranges": headers.get("Accept-Ranges"),
            }
    except Exception as exc:  # network metadata failure is reported, not hidden
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "content_length": None,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soma", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    soma_report = inspect_soma_csv(args.soma)
    shard_items = _list_public_gcs_objects(H01_BUCKET, SYNAPSE_PREFIX)
    shard_report = summarize_object_metadata(shard_items, prefix=SYNAPSE_PREFIX)
    crest_report = _head_metadata(CREST_URL)
    selected_source = choose_connectivity_source(
        crest_report.get("content_length"), crest_limit_bytes=CREST_LIMIT_BYTES
    )

    report = {
        "experiment": "006-stage-a",
        "preregistration_issue": 13,
        "outcome_blind": True,
        "forbidden_metrics_computed": False,
        "soma": soma_report,
        "connectivity_storage_metadata": {
            "h01_synapse_prefix": shard_report,
            "crest": crest_report,
            "crest_full_download_limit_bytes": CREST_LIMIT_BYTES,
            "selected_source_for_stage_b": selected_source,
            "selection_reason": "storage-feasibility-only",
        },
        "guardrail": (
            "No synapse shard or edge-table content was opened. This report contains only "
            "soma metadata and connectivity object/storage metadata permitted by Issue #13."
        ),
    }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
