from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pandas as pd
import pytest

import fcr.data.microns_roi_unit_bridge as bridge
from fcr.data.microns_roi_unit_bridge import (
    COREG_SOURCE,
    FROZEN_ASSET_PATH,
    RELEASE_COREG_COLUMNS,
    _git_blob_sha1,
    _read_allowed_nwb_value,
    cohort_csv_bytes,
    enforce_one_to_one_structural_cells,
    join_units_to_release,
    load_scanunit_dataframe,
    map_rois_to_units,
    parse_release_coregistration,
    scan_nwb_roi_identity,
)


class FakeDataset:
    def __init__(self, value: object):
        self.value = np.asarray(value)
        self.shape = self.value.shape
        self.dtype = self.value.dtype
        self.reads = 0

    def __getitem__(self, key: object) -> np.ndarray:
        self.reads += 1
        return self.value


def _roi(
    *,
    plane: str = "PlaneSegmentation2",
    mask_id: int = 7,
    field: int = 2,
) -> dict[str, object]:
    return {
        "asset_path": FROZEN_ASSET_PATH,
        "plane": plane,
        "roi_id": mask_id,
        "mask_id": mask_id,
        "session": 9,
        "scan_idx": 4,
        "field": field,
    }


def _release_bytes(rows: list[tuple[object, ...]]) -> bytes:
    header = ",".join(RELEASE_COREG_COLUMNS)
    body = [",".join(str(value) for value in row) for row in rows]
    return ("\n".join([header, *body]) + "\n").encode()


def test_value_read_guard_rejects_functional_arrays() -> None:
    functional = FakeDataset(np.array([0.1, 0.2]))
    with pytest.raises(RuntimeError, match="forbids NWB value read"):
        _read_allowed_nwb_value(
            functional,
            "/processing/ophys/Fluorescence/RoiResponseSeries1/data",
        )
    assert functional.reads == 0


def test_nwb_scan_reads_only_session_and_plane_ids() -> None:
    session_id = FakeDataset(np.array(b"9-scan-4"))
    plane2_id = FakeDataset(np.array([1, 2, 3], dtype=np.int64))
    plane4_id = FakeDataset(np.array([10, 11], dtype=np.int64))
    fluorescence = FakeDataset(np.zeros((100, 3), dtype=np.float32))
    timestamps = FakeDataset(np.arange(100, dtype=np.float64))

    h5 = {
        "session_id": session_id,
        "processing": {
            "ophys": {
                "ImageSegmentation": {
                    "PlaneSegmentation2": {"id": plane2_id},
                    "PlaneSegmentation4": {"id": plane4_id},
                },
                "Fluorescence": {
                    "RoiResponseSeries1": {
                        "data": fluorescence,
                        "timestamps": timestamps,
                    }
                },
            }
        },
    }

    report, rows = scan_nwb_roi_identity(h5)
    assert report["functional_values_read"] is False
    assert report["session"] == 9
    assert report["scan_idx"] == 4
    assert report["total_roi_rows"] == 5
    assert {(row["field"], row["mask_id"]) for row in rows} == {
        (2, 1),
        (2, 2),
        (2, 3),
        (4, 10),
        (4, 11),
    }
    assert session_id.reads == 1
    assert plane2_id.reads == 1
    assert plane4_id.reads == 1
    assert fluorescence.reads == 0
    assert timestamps.reads == 0


def test_nwb_scan_rejects_wrong_session_and_nonconforming_plane() -> None:
    wrong_session = {
        "session_id": FakeDataset(np.array(b"8-scan-4")),
        "processing": {"ophys": {"ImageSegmentation": {}}},
    }
    with pytest.raises(RuntimeError, match="contradicts frozen"):
        scan_nwb_roi_identity(wrong_session)

    bad_plane = {
        "session_id": FakeDataset(np.array(b"9-scan-4")),
        "processing": {
            "ophys": {"ImageSegmentation": {"PlaneSegmentationX": {"id": FakeDataset([1])}}}
        },
    }
    with pytest.raises(RuntimeError, match="non-conforming"):
        scan_nwb_roi_identity(bad_plane)


def test_scanunit_is_verified_before_pickle_deserialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {"session": [9], "scan_idx": [4], "unit_id": [17], "field": [2], "mask_id": [7]}
    )
    data = pickle.dumps(frame, protocol=4)
    monkeypatch.setattr(bridge, "SCANUNIT_SIZE_BYTES", len(data))
    monkeypatch.setattr(bridge, "SCANUNIT_GIT_BLOB_SHA1", _git_blob_sha1(data))

    loaded, report = load_scanunit_dataframe(data)
    assert loaded.equals(frame)
    assert report["size_bytes"] == len(data)
    assert report["sha256"] == hashlib.sha256(data).hexdigest()

    called = False

    def forbidden_read_pickle(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("pickle must not be opened after provenance failure")

    monkeypatch.setattr(bridge.pd, "read_pickle", forbidden_read_pickle)
    monkeypatch.setattr(bridge, "SCANUNIT_GIT_BLOB_SHA1", "0" * 40)
    with pytest.raises(RuntimeError, match="Git blob changed"):
        load_scanunit_dataframe(data)
    assert called is False


def test_scanunit_exact_join_reports_missing_and_rejects_conflicts() -> None:
    rois = [_roi(mask_id=1), _roi(mask_id=2), _roi(mask_id=3)]
    frame = pd.DataFrame(
        {
            "session": [9, 9, 8],
            "scan_idx": [4, 4, 4],
            "field": [2, 2, 2],
            "mask_id": [1, 2, 3],
            "unit_id": [101, 102, 999],
        }
    )
    mapped, report = map_rois_to_units(rois, frame)
    assert [row["unit_id"] for row in mapped] == [101, 102]
    assert report["exact_match_roi_rows"] == 2
    assert report["unmapped_roi_rows"] == 1

    conflict = pd.DataFrame(
        {
            "session": [9, 9],
            "scan_idx": [4, 4],
            "field": [2, 2],
            "mask_id": [1, 1],
            "unit_id": [101, 202],
        }
    )
    with pytest.raises(RuntimeError, match="conflicting target identity keys"):
        map_rois_to_units([_roi(mask_id=1)], conflict)


def test_release_parser_requires_frozen_schema_and_exact_positive_root() -> None:
    good = _release_bytes(
        [
            (1, "t", 10, 11, 12, 111, 1001, 9, 4, 101),
            (2, "t", 20, 21, 22, 222, 2002, 8, 4, 999),
        ]
    )
    mapping, report = parse_release_coregistration(good)
    assert mapping[101] == {(1001, 10, 11, 12)}
    assert report["target_session_scan_rows"] == 1

    wrong_schema = good.replace(b"pt_root_id", b"root_id", 1)
    with pytest.raises(RuntimeError, match="unexpected func_unit_em_match_release schema"):
        parse_release_coregistration(wrong_schema)

    zero_root = _release_bytes([(1, "t", 10, 11, 12, 111, 0, 9, 4, 101)])
    with pytest.raises(RuntimeError, match="below minimum 1"):
        parse_release_coregistration(zero_root)


def test_release_membership_is_deterministic_and_ambiguous_unit_is_not_chosen() -> None:
    unit_rows = [
        {**_roi(mask_id=1), "unit_id": 101},
        {**_roi(mask_id=2), "unit_id": 102},
        {**_roi(mask_id=3), "unit_id": 103},
    ]
    release = {
        101: {(1001, 10, 11, 12)},
        102: {(2001, 20, 21, 22), (2002, 23, 24, 25)},
    }
    rows, report = join_units_to_release(unit_rows, release)
    assert len(rows) == 1
    assert rows[0]["unit_id"] == 101
    assert rows[0]["v117_pt_root_id"] == 1001
    assert rows[0]["coreg_source"] == COREG_SOURCE
    assert report["unmatched_release_roi_rows"] == 1
    assert report["ambiguous_release_units"] == 1
    assert report["ambiguous_release_roi_rows"] == 1


def test_duplicate_structural_root_excludes_all_duplicate_roi_rows() -> None:
    rows = [
        {
            **_roi(mask_id=1),
            "unit_id": 101,
            "v117_pt_root_id": 9001,
            "pt_position_x": 1,
            "pt_position_y": 2,
            "pt_position_z": 3,
            "coreg_source": COREG_SOURCE,
        },
        {
            **_roi(plane="PlaneSegmentation4", mask_id=2, field=4),
            "unit_id": 102,
            "v117_pt_root_id": 9001,
            "pt_position_x": 1,
            "pt_position_y": 2,
            "pt_position_z": 3,
            "coreg_source": COREG_SOURCE,
        },
        {
            **_roi(mask_id=3),
            "unit_id": 103,
            "v117_pt_root_id": 9002,
            "pt_position_x": 4,
            "pt_position_y": 5,
            "pt_position_z": 6,
            "coreg_source": COREG_SOURCE,
        },
    ]
    kept, report = enforce_one_to_one_structural_cells(rows)
    assert [row["v117_pt_root_id"] for row in kept] == [9002]
    assert report["duplicate_structural_roots"] == 1
    assert report["roi_rows_excluded_for_duplicate_root"] == 2


def test_manifest_is_deterministic() -> None:
    rows = [
        {
            **_roi(plane="PlaneSegmentation4", mask_id=8, field=4),
            "unit_id": 202,
            "v117_pt_root_id": 2002,
            "pt_position_x": 20,
            "pt_position_y": 21,
            "pt_position_z": 22,
            "coreg_source": COREG_SOURCE,
        },
        {
            **_roi(mask_id=7),
            "unit_id": 101,
            "v117_pt_root_id": 1001,
            "pt_position_x": 10,
            "pt_position_y": 11,
            "pt_position_z": 12,
            "coreg_source": COREG_SOURCE,
        },
    ]
    assert cohort_csv_bytes(rows) == cohort_csv_bytes(list(reversed(rows)))
    assert b"release-func-unit-em-match-v117" in cohort_csv_bytes(rows)
