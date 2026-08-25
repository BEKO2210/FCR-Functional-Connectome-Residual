from __future__ import annotations

import numpy as np
import pytest

from fcr.data.microns_functional_preflight import (
    COREG_SOURCE,
    FROZEN_ASSET_PATH,
    STATIC_NUCLEUS_COLUMNS,
    _read_allowed_value,
    cohort_csv_bytes,
    decode_ragged_nucleus_ids,
    scan_hdf5_metadata,
    summarize_root_ids,
    validate_static_nucleus_table,
)


class FakeDataset:
    def __init__(
        self,
        value: object,
        *,
        shape: tuple[int, ...] | None = None,
        dtype: object = None,
    ):
        self.value = np.asarray(value)
        self.shape = self.value.shape if shape is None else shape
        self.dtype = self.value.dtype if dtype is None else np.dtype(dtype)
        self.reads = 0

    def __getitem__(self, key: object) -> np.ndarray:
        self.reads += 1
        return self.value


def test_large_integer_root_ids_remain_exact() -> None:
    values = np.array([0, 864691135737000001, 864691135737000129], dtype=np.uint64)
    result = summarize_root_ids(values)
    assert result.precision_safe is True
    assert result.normalized == (864691135737000001, 864691135737000129)


def test_large_float_root_ids_are_rejected_as_unsafe_identity() -> None:
    values = np.array([np.nan, float(2**53 + 128), float(2**53 + 256)])
    result = summarize_root_ids(values)
    assert result.precision_safe is False
    assert result.normalized is None


def test_ragged_cave_ids_decode_without_choosing_ambiguous_rows() -> None:
    values = np.array([np.nan, 101.0, 102.0, 103.0, 104.0])
    index = np.array([1, 2, 4, 5], dtype=np.uint16)
    result = decode_ragged_nucleus_ids(values, index, row_count=4)
    assert result.precision_safe is True
    assert result.row_nucleus_ids == (None, 101, None, 104)
    assert result.row_is_ambiguous == (False, False, True, False)


def test_value_read_guard_rejects_functional_arrays() -> None:
    functional = FakeDataset(np.array([0.1, 0.2]))
    with pytest.raises(RuntimeError, match="forbids value reads"):
        _read_allowed_value(
            functional,
            "/processing/ophys/Fluorescence/RoiResponseSeries1/data",
        )
    assert functional.reads == 0


def test_cohort_manifest_is_deterministic_and_contains_frozen_fields() -> None:
    rows = [
        {
            "asset_path": "asset",
            "plane": "PlaneSegmentation2",
            "roi_id": "2",
            "legacy_cave_id": 22,
            "pt_position_x": 7,
            "pt_position_y": 8,
            "pt_position_z": 9,
            "nucleus_id": 202,
            "v117_pt_root_id": 2000,
            "coreg_source": COREG_SOURCE,
        },
        {
            "asset_path": "asset",
            "plane": "PlaneSegmentation1",
            "roi_id": "1",
            "legacy_cave_id": 11,
            "pt_position_x": 1,
            "pt_position_y": 2,
            "pt_position_z": 3,
            "nucleus_id": 101,
            "v117_pt_root_id": 1000,
            "coreg_source": COREG_SOURCE,
        },
    ]
    encoded = cohort_csv_bytes(rows)
    assert encoded == cohort_csv_bytes(list(reversed(rows)))
    assert b"legacy_cave_id" in encoded
    assert b"pt_position_x" in encoded


def test_static_nucleus_table_joins_by_exact_position_not_legacy_id() -> None:
    candidate_rows = [
        {
            "asset_path": "asset",
            "plane": "PlaneSegmentation2",
            "roi_id": "7",
            "legacy_cave_id": 999999,
            "pt_position_x": 1,
            "pt_position_y": 2,
            "pt_position_z": 3,
            "coreg_source": COREG_SOURCE,
        },
        {
            "asset_path": "asset",
            "plane": "PlaneSegmentation4",
            "roi_id": "8",
            "legacy_cave_id": 888888,
            "pt_position_x": 7,
            "pt_position_y": 8,
            "pt_position_z": 9,
            "coreg_source": COREG_SOURCE,
        },
    ]
    raw = (
        b"101,t,111,864691135737000001,1,2,3,100.0\n"
        b"102,t,112,864691135737000002,4,5,6,101.0\n"
        b"103,t,113,864691135737000003,7,8,9,102.0\n"
    )
    rows, report = validate_static_nucleus_table(raw, candidate_rows)
    assert report["validation_ok"] is True
    assert report["missing_candidate_points"] == 0
    assert [row["nucleus_id"] for row in rows] == [101, 103]
    assert [row["v117_pt_root_id"] for row in rows] == [
        864691135737000001,
        864691135737000003,
    ]
    assert report["column_order"] == list(STATIC_NUCLEUS_COLUMNS)


def test_static_position_join_hard_stops_on_missing_or_nonunique_point() -> None:
    candidate = [
        {
            "asset_path": "asset",
            "plane": "PlaneSegmentation2",
            "roi_id": "7",
            "legacy_cave_id": 1,
            "pt_position_x": 1,
            "pt_position_y": 2,
            "pt_position_z": 3,
            "coreg_source": COREG_SOURCE,
        }
    ]
    _, missing = validate_static_nucleus_table(
        b"101,t,111,1,4,5,6,100.0\n", candidate
    )
    assert missing["validation_ok"] is False
    assert missing["missing_candidate_points"] == 1

    duplicate_raw = (
        b"101,t,111,1,1,2,3,100.0\n"
        b"102,t,112,2,1,2,3,101.0\n"
    )
    _, duplicate = validate_static_nucleus_table(duplicate_raw, candidate)
    assert duplicate["validation_ok"] is False
    assert duplicate["nonunique_candidate_points"] == 1


def test_static_join_excludes_zero_root_and_duplicate_canonical_nucleus() -> None:
    candidates = [
        {
            "asset_path": "asset",
            "plane": "PlaneSegmentation2",
            "roi_id": "1",
            "legacy_cave_id": 10,
            "pt_position_x": 1,
            "pt_position_y": 2,
            "pt_position_z": 3,
            "coreg_source": COREG_SOURCE,
        },
        {
            "asset_path": "asset",
            "plane": "PlaneSegmentation2",
            "roi_id": "2",
            "legacy_cave_id": 20,
            "pt_position_x": 4,
            "pt_position_y": 5,
            "pt_position_z": 6,
            "coreg_source": COREG_SOURCE,
        },
        {
            "asset_path": "asset",
            "plane": "PlaneSegmentation4",
            "roi_id": "3",
            "legacy_cave_id": 30,
            "pt_position_x": 7,
            "pt_position_y": 8,
            "pt_position_z": 9,
            "coreg_source": COREG_SOURCE,
        },
    ]
    raw = (
        b"101,t,0,0,1,2,3,100.0\n"
        b"202,t,112,22,4,5,6,101.0\n"
        b"202,t,113,33,7,8,9,102.0\n"
    )
    rows, report = validate_static_nucleus_table(raw, candidates)
    assert rows == []
    assert report["candidate_points_without_positive_v117_root"] == 1
    assert report["duplicate_nucleus_identities"] == 1
    assert report["roi_rows_excluded_for_duplicate_nucleus"] == 2
    assert report["validation_ok"] is False


def test_scan_reads_positions_and_structural_ids_but_not_functional_values() -> None:
    root = FakeDataset(np.array([float(2**53 + 128), float(2**53 + 256), np.nan]))
    roi = FakeDataset(np.array([1, 2, 3], dtype=np.int64))
    cave = FakeDataset(np.array([101.0, 201.0, 202.0, np.nan]))
    cave_index = FakeDataset(np.array([1, 3, 4], dtype=np.uint16))
    pos_x = FakeDataset(np.array([10.0, 20.0, np.nan]))
    pos_y = FakeDataset(np.array([11.0, 21.0, np.nan]))
    pos_z = FakeDataset(np.array([12.0, 22.0, np.nan]))
    mask = FakeDataset(np.zeros((3, 2, 2), dtype=np.float32))
    fluorescence_data = FakeDataset(np.zeros((100, 3), dtype=np.float32))
    fluorescence_timestamps = FakeDataset(np.arange(100, dtype=np.float64))
    clip_start = FakeDataset(np.arange(5, dtype=np.float64))
    clip_ids = FakeDataset(np.arange(5, dtype=np.int64))

    h5 = {
        "identifier": FakeDataset(np.array(b"synthetic-nwb")),
        "session_description": FakeDataset(np.array(b"metadata-only")),
        "processing": {
            "ophys": {
                "ImageSegmentation": {
                    "PlaneSegmentation2": {
                        "id": roi,
                        "pt_root_id": root,
                        "cave_ids": cave,
                        "cave_ids_index": cave_index,
                        "pt_x_position": pos_x,
                        "pt_y_position": pos_y,
                        "pt_z_position": pos_z,
                        "image_mask": mask,
                    }
                },
                "Fluorescence": {
                    "RoiResponseSeries1": {
                        "data": fluorescence_data,
                        "timestamps": fluorescence_timestamps,
                    }
                },
            }
        },
        "intervals": {"Clip": {"id": clip_ids, "start_time": clip_start}},
    }
    report, candidates = scan_hdf5_metadata(h5, asset_path=FROZEN_ASSET_PATH)
    coreg = report["coregistration"]
    assert report["functional_values_read"] is False
    assert coreg["canonical_identity"] == "nucleus_detection_v0.id"
    assert coreg["coreg_source"] == COREG_SOURCE
    assert coreg["nwb_pt_root_id_exact_integer_safe"] is False
    # Row 1 is retained, row 2 is excluded because its legacy CAVE ragged value is ambiguous,
    # row 3 has no structural position.
    assert len(candidates) == 1
    assert candidates[0]["legacy_cave_id"] == 101
    assert candidates[0]["pt_position_x"] == 10
    assert root.reads == 1
    assert roi.reads == 1
    assert cave.reads == 1
    assert cave_index.reads == 1
    assert pos_x.reads == 1
    assert pos_y.reads == 1
    assert pos_z.reads == 1
    assert mask.reads == 0
    assert fluorescence_data.reads == 0
    assert fluorescence_timestamps.reads == 0
    assert clip_start.reads == 0
    assert clip_ids.reads == 0


def test_scan_rejects_partial_or_fractional_structural_position() -> None:
    def make_h5(x: float, y: float, z: float) -> dict[str, object]:
        return {
            "processing": {
                "ophys": {
                    "ImageSegmentation": {
                        "PlaneSegmentation2": {
                            "id": FakeDataset(np.array([1], dtype=np.int64)),
                            "cave_ids": FakeDataset(np.array([101.0])),
                            "cave_ids_index": FakeDataset(np.array([1], dtype=np.uint16)),
                            "pt_x_position": FakeDataset(np.array([x])),
                            "pt_y_position": FakeDataset(np.array([y])),
                            "pt_z_position": FakeDataset(np.array([z])),
                        }
                    }
                }
            }
        }

    with pytest.raises(RuntimeError, match="partial v117 point"):
        scan_hdf5_metadata(make_h5(1.0, np.nan, 3.0), asset_path="asset")
    with pytest.raises(RuntimeError, match="not exact non-negative integer"):
        scan_hdf5_metadata(make_h5(1.5, 2.0, 3.0), asset_path="asset")
