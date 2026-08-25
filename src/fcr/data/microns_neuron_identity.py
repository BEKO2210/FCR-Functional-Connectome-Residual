"""Verified identity-only bridge for MICrONS Experiment 012.

This module is intentionally limited to the official ``Neuron.pkl`` identity
object and the already frozen public-v185 structural graph. It does not open
calcium traces, tuning objects, stimulus data, or FCR model outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import pickle
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .microns_public_l23 import load_public_l23_candidate_data

EXPERIMENT = "012"
PREREGISTRATION_ISSUE = 33
EVIDENCE_LEVEL = "E0-functional-structural-identity-bridge"

ALLEN_REFERENCE_COMMIT = "3f53e35e2bfd3063469dcbab0e1dedce5a82e3ca"
FUNCTIONAL_ARCHIVE_URL = (
    "https://zenodo.org/record/6363348/files/211019_vignette_functional_analysis_data.tgz"
)
FUNCTIONAL_ARCHIVE_MD5 = "bcb0d4f678909fbd481ac0b01242ae5c"
FUNCTIONAL_ARCHIVE_SHA256 = "5ee95299b1955832f3f7b07defcea842f95bdd0bb351cb9828581f96b96d868b"
NESTED_MEMBER = "function_data_tables.tgz"
NESTED_MEMBER_SIZE = 79_195_487
NESTED_MEMBER_SHA256 = "be972c2468bb4c32cca508870da02082697d9f3a37ce649743c21ce6c5f7a5b4"
NEURON_MEMBER = "Neuron.pkl"
NEURON_MEMBER_SIZE = 9_009
NEURON_MEMBER_SHA256 = "0327f71c3dfa85139a2185964d69b868e7e2370b260315baeb999a915052cdc1"

SOMA_URL = "https://zenodo.org/records/7510511/files/soma_valence_v185.csv?download=1"
SYNAPSE_URL = (
    "https://zenodo.org/records/7510511/files/"
    "soma_subgraph_synapses_spines_v185.csv?download=1"
)
SOMA_MD5 = "b7c349b1ecbdad46185bb98d88b96d20"
SYNAPSE_MD5 = "5bbb8ff59dcad4ccea6d46930920cd04"
EXPECTED_STRUCTURAL_NODES = 334

_UINT64_MAX = (1 << 64) - 1


class _RestrictedNeuronUnpickler(pickle.Unpickler):
    """Unpickle the pinned NumPy identity object with a tiny global allowlist."""

    _ALLOWED_GLOBALS = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"blocked pickle global: {module}.{name}")
        return super().find_class(module, name)


def _hash_md5(raw: bytes) -> str:
    return hashlib.md5(raw, usedforsecurity=False).hexdigest()


def _hash_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_exact_tar_member(raw: bytes, member_name: str, *, label: str) -> tuple[bytes, int]:
    try:
        archive = tarfile.open(fileobj=io.BytesIO(raw), mode="r:*")
    except tarfile.TarError as exc:
        raise RuntimeError(f"{label} is not a valid tar archive") from exc

    with archive:
        matches = [member for member in archive.getmembers() if member.name == member_name]
        if len(matches) != 1:
            raise RuntimeError(
                f"{label} must contain exactly one {member_name!r}; found {len(matches)}"
            )
        member = matches[0]
        if not member.isfile():
            raise RuntimeError(f"{member_name!r} is not a regular file")
        handle = archive.extractfile(member)
        if handle is None:
            raise RuntimeError(f"could not read {member_name!r}")
        payload = handle.read()
        if len(payload) != member.size:
            raise RuntimeError(f"size mismatch while reading {member_name!r}")
        return payload, int(member.size)


def extract_verified_neuron_pickle(functional_archive_raw: bytes) -> tuple[bytes, dict[str, object]]:
    """Verify the frozen archive chain and return only ``Neuron.pkl`` bytes."""
    observed_md5 = _hash_md5(functional_archive_raw)
    observed_sha256 = _hash_sha256(functional_archive_raw)
    if observed_md5 != FUNCTIONAL_ARCHIVE_MD5:
        raise RuntimeError("functional archive MD5 mismatch")
    if observed_sha256 != FUNCTIONAL_ARCHIVE_SHA256:
        raise RuntimeError("functional archive SHA-256 mismatch")

    nested_raw, nested_size = _read_exact_tar_member(
        functional_archive_raw,
        NESTED_MEMBER,
        label="functional archive",
    )
    if nested_size != NESTED_MEMBER_SIZE:
        raise RuntimeError("nested functional archive size mismatch")
    if _hash_sha256(nested_raw) != NESTED_MEMBER_SHA256:
        raise RuntimeError("nested functional archive SHA-256 mismatch")

    neuron_raw, neuron_size = _read_exact_tar_member(
        nested_raw,
        NEURON_MEMBER,
        label="nested functional archive",
    )
    if neuron_size != NEURON_MEMBER_SIZE:
        raise RuntimeError("Neuron.pkl size mismatch")
    neuron_sha256 = _hash_sha256(neuron_raw)
    if neuron_sha256 != NEURON_MEMBER_SHA256:
        raise RuntimeError("Neuron.pkl SHA-256 mismatch")

    provenance = {
        "archive_url": FUNCTIONAL_ARCHIVE_URL,
        "archive_md5": observed_md5,
        "archive_sha256": observed_sha256,
        "archive_size_bytes": len(functional_archive_raw),
        "nested_member": NESTED_MEMBER,
        "nested_size_bytes": nested_size,
        "nested_sha256": NESTED_MEMBER_SHA256,
        "neuron_member": NEURON_MEMBER,
        "neuron_size_bytes": neuron_size,
        "neuron_sha256": neuron_sha256,
    }
    return neuron_raw, provenance


def _restricted_load_neuron(neuron_raw: bytes) -> Mapping[str, object]:
    try:
        value = _RestrictedNeuronUnpickler(io.BytesIO(neuron_raw)).load()
    except (pickle.UnpicklingError, EOFError, ValueError, TypeError) as exc:
        raise RuntimeError("Neuron.pkl failed restricted deserialization") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("Neuron.pkl top-level object is not a mapping")
    if "segment_id" not in value:
        raise RuntimeError("Neuron.pkl is missing segment_id")
    return value


def _exact_positive_uint64_sequence(values: object) -> np.ndarray:
    if isinstance(values, (str, bytes)) or not isinstance(values, (Sequence, np.ndarray)):
        raise RuntimeError("segment_id is not a one-dimensional sequence")

    array = np.asarray(values)
    if array.ndim != 1:
        raise RuntimeError("segment_id must be one-dimensional")
    if np.issubdtype(array.dtype, np.floating):
        raise RuntimeError("floating-point segment IDs are forbidden")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise RuntimeError("complex segment IDs are forbidden")

    normalized: list[int] = []
    for raw_value in array.tolist():
        if isinstance(raw_value, (bool, np.bool_)):
            raise RuntimeError("boolean segment ID is forbidden")
        if not isinstance(raw_value, (int, np.integer)):
            raise RuntimeError("segment IDs must be exact integers")
        value = int(raw_value)
        if value <= 0 or value > _UINT64_MAX:
            raise RuntimeError("segment ID is outside positive uint64 range")
        normalized.append(value)

    if len(normalized) != len(set(normalized)):
        raise RuntimeError("duplicate segment IDs are a hard stop")
    return np.asarray(normalized, dtype=np.uint64)


def load_verified_functional_segment_ids(neuron_raw: bytes) -> tuple[np.ndarray, list[str]]:
    """Read only the authorized ``segment_id`` value from verified ``Neuron.pkl`` bytes."""
    neuron = _restricted_load_neuron(neuron_raw)
    key_names = sorted(str(key) for key in neuron.keys())
    segment_ids = _exact_positive_uint64_sequence(neuron["segment_id"])
    return segment_ids, key_names


def _identity_hash(ids: list[int]) -> str:
    payload = "segment_id\n" + "".join(f"{value}\n" for value in ids)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def build_matched_manifest(matched_ids: list[int]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["segment_id"])
    for segment_id in matched_ids:
        writer.writerow([segment_id])
    return stream.getvalue().encode("ascii")


def compute_exact_identity_overlap(
    functional_segment_ids: np.ndarray,
    structural_node_ids: np.ndarray,
) -> dict[str, object]:
    """Compute the preregistered exact set intersection with no fuzzy matching."""
    functional = sorted(int(value) for value in functional_segment_ids.tolist())
    structural = sorted(int(value) for value in structural_node_ids.tolist())
    if len(structural) != EXPECTED_STRUCTURAL_NODES:
        raise RuntimeError(
            f"frozen structural graph must contain {EXPECTED_STRUCTURAL_NODES} nodes; "
            f"found {len(structural)}"
        )
    if len(structural) != len(set(structural)):
        raise RuntimeError("frozen structural node IDs are not unique")

    functional_set = set(functional)
    structural_set = set(structural)
    matched = sorted(functional_set & structural_set)
    functional_only = sorted(functional_set - structural_set)
    structural_only = sorted(structural_set - functional_set)
    manifest = build_matched_manifest(matched)

    return {
        "functional_identity_count": len(functional),
        "structural_identity_count": len(structural),
        "exact_overlap_count": len(matched),
        "functional_only_count": len(functional_only),
        "structural_only_count": len(structural_only),
        "functional_identity_sha256": _identity_hash(functional),
        "structural_identity_sha256": _identity_hash(structural),
        "matched_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "matched_ids": matched,
        "manifest_bytes": manifest,
    }


def run_identity_bridge(
    *,
    functional_archive: str | Path,
    soma_csv: str | Path,
    synapse_csv: str | Path,
) -> tuple[dict[str, object], bytes]:
    """Run the frozen Experiment-012 identity bridge."""
    archive_raw = Path(functional_archive).read_bytes()
    neuron_raw, functional_provenance = extract_verified_neuron_pickle(archive_raw)
    functional_ids, neuron_keys = load_verified_functional_segment_ids(neuron_raw)

    structural = load_public_l23_candidate_data(soma_csv, synapse_csv)
    overlap = compute_exact_identity_overlap(functional_ids, structural.node_ids)
    manifest_bytes = overlap.pop("manifest_bytes")
    overlap.pop("matched_ids")

    report: dict[str, object] = {
        "experiment": EXPERIMENT,
        "preregistration_issue": PREREGISTRATION_ISSUE,
        "evidence_level": EVIDENCE_LEVEL,
        "allen_reference_commit": ALLEN_REFERENCE_COMMIT,
        "functional_source": functional_provenance,
        "structural_source": {
            "soma_url": SOMA_URL,
            "soma_md5": SOMA_MD5,
            "synapse_url": SYNAPSE_URL,
            "synapse_md5": SYNAPSE_MD5,
            "graph_builder": "load_public_l23_candidate_data",
        },
        "neuron_top_level_keys": neuron_keys,
        "duplicate_functional_ids_seen": False,
        **overlap,
        "ease_trace_opened": False,
        "ease_tuning_opened": False,
        "stimulus_opened": False,
        "functional_measurements_interpreted": False,
        "connectivity_used_for_cohort_selection": False,
        "fcr_outcomes_used_for_cohort_selection": False,
        "identity_match_rule": "exact-segment-id-equality-only",
    }
    return report, manifest_bytes
