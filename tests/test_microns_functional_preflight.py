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
    assert result.non_null_count == 2
    assert result.unique_count == 2
    assert result.normalized == (864691135737000001, 864691135737000129)


def test_large_float_root_ids_are_rejected_as_unsafe_identity() -> None:
    values = np.array([np.nan, float(2**53 + 128), float(2**53 + 256)], dtype=np.float64)
    result = summarize_root_ids(values)

    assert result.precision_safe is False
    assert result.non_null_count == 2
    assert result.normalized is None


def test_ragged_cave_ids_decode_without_choosing_ambiguous_rows() -> None:
    values = np.array([np.nan, 101.0, 102.0, 103.0, 104.0], dtype=np.float64)
    index = np.array([1, 2, 4, 5], dtype=np.uint16)

    result = decode_ragged_nucleus_ids(values, index, row_count=4)

    assert result.precision_safe is True
    assert result.rows_unmatched == 1
    assert result.rows_single == 2
    assert result.rows_ambiguous == 1
    assert result.row_nucleus_ids == (None, 101, None, 104)


def test_value_read_guard_rejects_functional_arrays() -> None:
    functional = FakeDataset(np.array([0.1, 0.2]))

    with pytest.raises(RuntimeError, match="forbids value reads"):
        _read_allowed_value(
            functional,
            "/processing/ophys/Fluorescence/RoiResponseSeries1/data",
        )
    assert functional.reads == 0


def test_cohort_manifest_is_deterministic() -> None:
    rows = [
        {
            "asset_path": "asset",
            "plane": "PlaneSegmentation2",
            "roi_id": "2",
            "nucleus_id": 20,
            "coreg_source": COREG_SOURCE,
            "v117_pt_root_id": 2000,
        },
        {
            "asset_path": "asset",
            "plane": "PlaneSegmentation1",
            "roi_id": "1",
            "nucleus_id": 10,
            "coreg_source": COREG_SOURCE,
            "v117_pt_root_id": 1000,
        },
    ]

    assert cohort_csv_bytes(rows) == cohort_csv_bytes(list(reversed(rows)))


def test_static_nucleus_table_requires_headerless_exact_unique_matches() -> None:
    candidate_rows = [
        {
            "asset_path": "asset",
            "plane": "PlaneSegmentation2",
            "roi_id": "7",
            "nucleus_id": 101,
            "coreg_source": COREG_SOURCE,
        },
        {
            "asset_path": "asset",
            "plane": "PlaneSegmentation4",
            "roi_id": "8",
            "nucleus_id": 103,
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
    assert report["matched_unique_nucleus_ids"] == 2
    assert report["column_order"] == list(STATIC_NUCLEUS_COLUMNS)
    assert [row["v117_pt_root_id"] for row in rows] == [
        864691135737000001,
        864691135737000003,
    ]


def test_static_nucleus_table_excludes_zero_root_without_guessing() -> None:
    candidate_rows = [
        {
            "asset_path": "asset",
            "plane": "PlaneSegmentation2",
            "roi_id": "7",
            "nucleus_id": 101,
            "coreg_source": COREG_SOURCE,
        },
        {
            "asset_path": "asset",
            "plane": "PlaneSegmentation4",
            "roi_id": "8",
            "nucleus_id": 103,
            "coreg_source": COREG_SOURCE,
        },
    ]
    raw = (
        b"101,t,0,0,1,2,3,100.0\n"
        b"103,t,113,864691135737000003,7,8,9,102.0\n"
    )

    rows, report = validate_static_nucleus_table(raw, candidate_rows)

    assert report["validation_ok"] is True
    assert report["candidate_ids_without_positive_v117_root"] == 1
    assert report["roi_rows_excluded_without_positive_v117_root"] == 1
    assert [row["nucleus_id"] for row in rows] == [103]


def test_scan_reads_coregistration_but_not_functional_values() -> None:
    plane1_root = FakeDataset(
        np.array([np.nan, float(2**53 + 128), float(2**53 + 256)], dtype=np.float64)
    )
    plane1_roi = FakeDataset(np.array([1, 2, 3], dtype=np.int64))
    plane1_cave = FakeDataset(np.array([101.0, np.nan, 102.0], dtype=np.float64))
    plane1_index = FakeDataset(np.array([1, 2, 3], dtype=np.uint16))
    plane1_mask = FakeDataset(np.zeros((3, 2, 2), dtype=np.float32))

    plane2_root = FakeDataset(
        np.array([float(2**53 + 128), float(2**53 + 256)], dtype=np.float64)
    )
    plane2_roi = FakeDataset(np.array([4, 5], dtype=np.int64))
    plane2_cave = FakeDataset(np.array([101.0, 103.0], dtype=np.float64))
    plane2_index = FakeDataset(np.array([1, 2], dtype=np.uint16))
    plane2_mask = FakeDataset(np.zeros((2, 2, 2), dtype=np.float32))

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
                    "PlaneSegmentation1": {
                        "id": plane1_roi,
                        "pt_root_id": plane1_root,
                        "cave_ids": plane1_cave,
                        "cave_ids_index": plane1_index,
                        "image_mask": plane1_mask,
                    },
                    "PlaneSegmentation2": {
                        "id": plane2_roi,
                        "pt_root_id": plane2_root,
                        "cave_ids": plane2_cave,
                        "cave_ids_index": plane2_index,
                        "image_mask": plane2_mask,
                    },
                },
                "Fluorescence": {
                    "RoiResponseSeries1": {
                        "data": fluorescence_data,
                        "timestamps": fluorescence_timestamps,
                    }
                },
            }
        },
        "intervals": {
            "Clip": {
                "id": clip_ids,
                "start_time": clip_start,
            }
        },
    }

    report, candidates = scan_hdf5_metadata(h5, asset_path=FROZEN_ASSET_PATH)

    coreg = report["coregistration"]
    assert report["functional_values_read"] is False
    assert coreg["canonical_identity"] == "nucleus_id"
    assert coreg["nucleus_id_precision_safe"] is True
    assert coreg["nwb_pt_root_id_exact_integer_safe"] is False
    assert coreg["duplicate_nucleus_identities"] == 1
    assert coreg["roi_rows_excluded_for_duplicate_nucleus"] == 2
    assert {row["nucleus_id"] for row in candidates} == {102, 103}

    assert plane1_root.reads == 1
    assert plane1_roi.reads == 1
    assert plane1_cave.reads == 1
    assert plane1_index.reads == 1
    assert plane2_root.reads == 1
    assert plane2_roi.reads == 1
    assert plane2_cave.reads == 1
    assert plane2_index.reads == 1
    assert plane1_mask.reads == 0
    assert plane2_mask.reads == 0
    assert fluorescence_data.reads == 0
    assert fluorescence_timestamps.reads == 0
    assert clip_start.reads == 0
    assert clip_ids.reads == 0
