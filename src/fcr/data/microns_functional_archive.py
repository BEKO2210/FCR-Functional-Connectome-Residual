"""Outcome-blind inventory helpers for MICrONS Experiment 011.

This module inventories the official Layer 2/3 functional-vignette archive as
opaque bytes. It intentionally does not deserialize scientific data objects or
interpret functional measurements.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import PurePosixPath

EXPERIMENT = "011"
PREREGISTRATION_ISSUE = 31
EVIDENCE_LEVEL = "E0-functional-archive-inventory-only"

ALLEN_REFERENCE_COMMIT = "3f53e35e2bfd3063469dcbab0e1dedce5a82e3ca"
ARCHIVE_URL = (
    "https://zenodo.org/record/6363348/files/"
    "211019_vignette_functional_analysis_data.tgz"
)
PUBLISHED_MD5 = "bcb0d4f678909fbd481ac0b01242ae5c"

EXPECTED_DOCUMENTED_MEMBERS = {
    "Neuron.pkl",
    "EASETrace.pkl",
    "EASETuning.pkl",
}
IDENTITY_RELEVANT_BASENAMES = EXPECTED_DOCUMENTED_MEMBERS | {
    "function_data_tables.tgz",
}
SEPARATE_FUNCTIONAL_IDENTITY_BASENAMES = {
    "functional_identity.csv",
    "ease_identity.csv",
    "segment_scan.csv",
    "segment_scan_ids.csv",
}
NESTED_ARCHIVE_SUFFIXES = (".tgz", ".tar.gz", ".tar")


def _safe_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"unsafe archive member path: {name!r}")
    return normalized


def _member_type(member: tarfile.TarInfo) -> str:
    if member.isdir():
        return "dir"
    if member.isfile():
        return "file"
    raise RuntimeError(f"unsupported archive member type for {member.name!r}")


def _read_regular_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    if not member.isfile():
        raise RuntimeError(f"archive member is not a regular file: {member.name!r}")
    handle = archive.extractfile(member)
    if handle is None:
        raise RuntimeError(f"could not read archive member: {member.name!r}")
    return handle.read()


def _inventory_tar_bytes(
    raw: bytes,
    *,
    label: str,
    inspect_nested: bool,
) -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    try:
        archive = tarfile.open(fileobj=io.BytesIO(raw), mode="r:*")
    except tarfile.TarError as exc:
        raise RuntimeError(f"{label} is not a valid tar archive") from exc

    inventory: list[dict[str, object]] = []
    nested: dict[str, list[dict[str, object]]] = {}
    with archive:
        members = archive.getmembers()
        for member in members:
            safe_name = _safe_member_name(member.name)
            kind = _member_type(member)
            record: dict[str, object] = {
                "name": safe_name,
                "size_bytes": int(member.size),
                "type": kind,
            }
            if kind == "file":
                payload = _read_regular_member(archive, member)
                if len(payload) != member.size:
                    raise RuntimeError(f"archive member size mismatch: {safe_name!r}")
                record["sha256"] = hashlib.sha256(payload).hexdigest()
                lower_name = safe_name.lower()
                if inspect_nested and lower_name.endswith(NESTED_ARCHIVE_SUFFIXES):
                    nested_inventory, _ = _inventory_tar_bytes(
                        payload,
                        label=f"nested archive {safe_name}",
                        inspect_nested=False,
                    )
                    nested[safe_name] = nested_inventory
            inventory.append(record)

    inventory.sort(key=lambda row: str(row["name"]))
    for rows in nested.values():
        rows.sort(key=lambda row: str(row["name"]))
    return inventory, dict(sorted(nested.items()))


def _all_file_records(
    outer: list[dict[str, object]],
    nested: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    records = [row for row in outer if row["type"] == "file"]
    for parent, rows in nested.items():
        for row in rows:
            if row["type"] != "file":
                continue
            records.append(
                {
                    **row,
                    "name": f"{parent}::{row['name']}",
                }
            )
    return records


def _basename(record: dict[str, object]) -> str:
    name = str(record["name"]).split("::")[-1]
    return PurePosixPath(name).name


def inventory_functional_archive(raw: bytes) -> dict[str, object]:
    """Inventory the frozen MICrONS functional archive without interpreting values."""
    observed_md5 = hashlib.md5(raw, usedforsecurity=False).hexdigest()
    if observed_md5 != PUBLISHED_MD5:
        raise RuntimeError(
            "MICrONS functional archive MD5 mismatch: "
            f"expected {PUBLISHED_MD5}, got {observed_md5}"
        )

    outer, nested = _inventory_tar_bytes(
        raw,
        label="MICrONS functional archive",
        inspect_nested=True,
    )
    records = _all_file_records(outer, nested)
    basenames = {_basename(row) for row in records}

    identity_hashes = {
        str(row["name"]): str(row["sha256"])
        for row in records
        if _basename(row) in IDENTITY_RELEVANT_BASENAMES and "sha256" in row
    }
    documented_present = sorted(EXPECTED_DOCUMENTED_MEMBERS & basenames)
    documented_missing = sorted(EXPECTED_DOCUMENTED_MEMBERS - basenames)
    separate_identity = sorted(SEPARATE_FUNCTIONAL_IDENTITY_BASENAMES & basenames)

    if documented_missing:
        classification = "hard-stop"
    elif separate_identity:
        classification = "identity-source-available"
    else:
        classification = "identity-source-entangled-with-functional-values"

    return {
        "experiment": EXPERIMENT,
        "preregistration_issue": PREREGISTRATION_ISSUE,
        "evidence_level": EVIDENCE_LEVEL,
        "functional_values_deserialized": False,
        "functional_measurements_interpreted": False,
        "archive": {
            "url": ARCHIVE_URL,
            "published_md5": PUBLISHED_MD5,
            "observed_md5": observed_md5,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        },
        "allen_reference_commit": ALLEN_REFERENCE_COMMIT,
        "outer_inventory": outer,
        "nested_inventory": nested,
        "identity_relevant_member_sha256": dict(sorted(identity_hashes.items())),
        "documented_identity_members_present": documented_present,
        "documented_identity_members_missing": documented_missing,
        "separate_functional_identity_members": separate_identity,
        "segment_id_scan_id_available_without_functional_object_open": bool(separate_identity),
        "next_step_classification": classification,
    }
