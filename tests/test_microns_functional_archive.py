from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

import fcr.data.microns_functional_archive as archive_module
from fcr.data.microns_functional_archive import _inventory_tar_bytes, inventory_functional_archive


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def _run_synthetic(
    monkeypatch: pytest.MonkeyPatch,
    inner_files: dict[str, bytes],
) -> dict[str, object]:
    inner = _tar_bytes(inner_files)
    outer = _tar_bytes({"function_data_tables.tgz": inner, "README.txt": b"opaque"})
    md5 = hashlib.md5(outer, usedforsecurity=False).hexdigest()
    monkeypatch.setattr(archive_module, "PUBLISHED_MD5", md5)
    return inventory_functional_archive(outer)


def test_nested_inventory_reports_entangled_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _run_synthetic(
        monkeypatch,
        {
            "Neuron.pkl": b"opaque-neuron",
            "EASETrace.pkl": b"opaque-trace",
            "EASETuning.pkl": b"opaque-tuning",
        },
    )

    assert report["functional_values_deserialized"] is False
    assert report["functional_measurements_interpreted"] is False
    assert report["documented_identity_members_missing"] == []
    assert report["separate_functional_identity_members"] == []
    assert report["segment_id_scan_id_available_without_functional_object_open"] is False
    assert report["next_step_classification"] == "identity-source-entangled-with-functional-values"
    hashes = report["identity_relevant_member_sha256"]
    assert any(name.endswith("::Neuron.pkl") for name in hashes)
    assert any(name.endswith("::EASETrace.pkl") for name in hashes)
    assert any(name.endswith("::EASETuning.pkl") for name in hashes)


def test_separate_identity_table_changes_only_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _run_synthetic(
        monkeypatch,
        {
            "Neuron.pkl": b"opaque-neuron",
            "EASETrace.pkl": b"opaque-trace",
            "EASETuning.pkl": b"opaque-tuning",
            "segment_scan_ids.csv": b"segment_id,scan_id\n1,2\n",
        },
    )

    assert report["separate_functional_identity_members"] == ["segment_scan_ids.csv"]
    assert report["segment_id_scan_id_available_without_functional_object_open"] is True
    assert report["next_step_classification"] == "identity-source-available"


def test_missing_documented_member_is_hard_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _run_synthetic(
        monkeypatch,
        {
            "Neuron.pkl": b"opaque-neuron",
            "EASETrace.pkl": b"opaque-trace",
        },
    )

    assert report["documented_identity_members_missing"] == ["EASETuning.pkl"]
    assert report["next_step_classification"] == "hard-stop"


def test_md5_mismatch_stops_before_inventory() -> None:
    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        inventory_functional_archive(_tar_bytes({"README.txt": b"x"}))


def test_path_traversal_is_rejected() -> None:
    raw = _tar_bytes({"../escape.txt": b"no"})
    with pytest.raises(RuntimeError, match="unsafe archive member path"):
        _inventory_tar_bytes(raw, label="synthetic", inspect_nested=False)


def test_absolute_path_is_rejected() -> None:
    raw = _tar_bytes({"/escape.txt": b"no"})
    with pytest.raises(RuntimeError, match="unsafe archive member path"):
        _inventory_tar_bytes(raw, label="synthetic", inspect_nested=False)


def test_symlink_member_is_rejected() -> None:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        info = tarfile.TarInfo(name="link")
        info.type = tarfile.SYMTYPE
        info.linkname = "target"
        archive.addfile(info)
    with pytest.raises(RuntimeError, match="unsupported archive member type"):
        _inventory_tar_bytes(stream.getvalue(), label="synthetic", inspect_nested=False)


def test_inventory_order_is_deterministic() -> None:
    raw = _tar_bytes({"z.txt": b"z", "a.txt": b"a", "m.txt": b"m"})
    inventory, nested = _inventory_tar_bytes(raw, label="synthetic", inspect_nested=False)
    assert [row["name"] for row in inventory] == ["a.txt", "m.txt", "z.txt"]
    assert nested == {}
