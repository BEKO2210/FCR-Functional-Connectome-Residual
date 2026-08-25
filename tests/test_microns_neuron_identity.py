from __future__ import annotations

import hashlib
import io
import pickle
import tarfile

import numpy as np
import pytest

import fcr.data.microns_neuron_identity as identity


def _neuron_pickle(values: np.ndarray) -> bytes:
    return pickle.dumps({"segment_id": values}, protocol=4)


def _tar_bytes(name: str, payload: bytes, *, gzip: bool = True) -> bytes:
    stream = io.BytesIO()
    mode = "w:gz" if gzip else "w"
    with tarfile.open(fileobj=stream, mode=mode) as archive:
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def test_verified_archive_chain_extracts_only_neuron_member(monkeypatch: pytest.MonkeyPatch) -> None:
    neuron = _neuron_pickle(np.asarray([11, 22, 33], dtype=np.uint64))
    nested = _tar_bytes("Neuron.pkl", neuron)
    outer = _tar_bytes("function_data_tables.tgz", nested)

    monkeypatch.setattr(identity, "FUNCTIONAL_ARCHIVE_MD5", hashlib.md5(outer).hexdigest())
    monkeypatch.setattr(identity, "FUNCTIONAL_ARCHIVE_SHA256", hashlib.sha256(outer).hexdigest())
    monkeypatch.setattr(identity, "NESTED_MEMBER_SIZE", len(nested))
    monkeypatch.setattr(identity, "NESTED_MEMBER_SHA256", hashlib.sha256(nested).hexdigest())
    monkeypatch.setattr(identity, "NEURON_MEMBER_SIZE", len(neuron))
    monkeypatch.setattr(identity, "NEURON_MEMBER_SHA256", hashlib.sha256(neuron).hexdigest())

    extracted, provenance = identity.extract_verified_neuron_pickle(outer)
    assert extracted == neuron
    assert provenance["neuron_size_bytes"] == len(neuron)


def test_restricted_loader_reads_exact_uint64_segment_ids() -> None:
    raw = _neuron_pickle(np.asarray([7, 9, 12], dtype=np.uint64))
    values, keys = identity.load_verified_functional_segment_ids(raw)
    assert values.dtype == np.uint64
    assert values.tolist() == [7, 9, 12]
    assert keys == ["segment_id"]


def test_restricted_loader_blocks_unapproved_pickle_global() -> None:
    class Dangerous:
        def __reduce__(self):
            return (eval, ("40 + 2",))

    raw = pickle.dumps({"segment_id": Dangerous()}, protocol=4)
    with pytest.raises(RuntimeError, match="restricted deserialization"):
        identity.load_verified_functional_segment_ids(raw)


def test_float_segment_ids_are_forbidden() -> None:
    raw = _neuron_pickle(np.asarray([7.0, 9.0], dtype=np.float64))
    with pytest.raises(RuntimeError, match="floating-point"):
        identity.load_verified_functional_segment_ids(raw)


def test_duplicate_segment_ids_are_hard_stop() -> None:
    raw = _neuron_pickle(np.asarray([7, 7, 9], dtype=np.uint64))
    with pytest.raises(RuntimeError, match="duplicate"):
        identity.load_verified_functional_segment_ids(raw)


def test_exact_overlap_is_deterministic() -> None:
    structural = np.arange(1, identity.EXPECTED_STRUCTURAL_NODES + 1, dtype=np.int64)
    functional = np.asarray([335, 2, 1], dtype=np.uint64)

    result = identity.compute_exact_identity_overlap(functional, structural)
    assert result["functional_identity_count"] == 3
    assert result["structural_identity_count"] == 334
    assert result["exact_overlap_count"] == 2
    assert result["functional_only_count"] == 1
    assert result["structural_only_count"] == 332
    assert result["matched_ids"] == [1, 2]
    assert result["manifest_bytes"] == b"segment_id\n1\n2\n"
    assert result["matched_manifest_sha256"] == hashlib.sha256(b"segment_id\n1\n2\n").hexdigest()


def test_structural_graph_count_is_frozen() -> None:
    with pytest.raises(RuntimeError, match="334 nodes"):
        identity.compute_exact_identity_overlap(
            np.asarray([1], dtype=np.uint64),
            np.asarray([1, 2, 3], dtype=np.int64),
        )
