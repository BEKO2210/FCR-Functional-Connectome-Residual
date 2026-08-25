from __future__ import annotations

import gzip
import hashlib

import numpy as np
import pytest

from fcr.data.microns_functional_preflight import (
    COREG_SOURCE,
    FROZEN_ASSET_PATH,
    V661_LOOKUP_HEADER_SHA256,
    V661_LOOKUP_SCHEMA,
    _read_allowed_value,
    cohort_csv_bytes,
    decode_ragged_nucleus_ids,
    parse_v661_lookup_header,
    scan_hdf5_metadata,
    summarize_root_ids,
    validate_v661_lookup_table,
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


def _frozen_header_bytes() -> bytes:
    # The archived GCS header uses CRLF; byte identity is intentional.
    return (
        "id,int64\r\n"
        "volume,float64\r\n"
        "pt_root_id,int64\r\n"
        "orig_root_id,int64\r\n"
        "pt_supervoxel_id,int64\r\n"
        "pt_position_x,int64\r\n"
        "pt_position_y,int64\r\n"
        "pt_position_z,int64\r\n"
        "pt_position_lookup_x,int64\r\n"
        "pt_position_lookup_y,int64\r\n"
        "pt_position_lookup_z,int64\r\n"
    ).encode("utf-8")


def _columns() -> tuple[str, ...]:
    return tuple(column for column, _dtype in V661_LOOKUP_SCHEMA)


def _candidate(
    roi_id: str,
    point: tuple[int, int, int],
    *,
    legacy_cave_id: int = 999,
) -> dict[str, object]:
    return {
        "asset_path": "asset",
        "plane": "PlaneSegmentation2",
        "roi_id": roi_id,
        "legacy_cave_id": legacy_cave_id,
        "pt_position_x": point[0],
        "pt_position_y": point[1],
        "pt_position_z": point[2],
        "coreg_source": COREG_SOURCE,
    }


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


def test_frozen_v661_header_requires_exact_bytes_and_schema() -> None:
    raw = _frozen_header_bytes()
    assert hashlib.sha256(raw).hexdigest() == V661_LOOKUP_HEADER_SHA256
    assert parse_v661_lookup_header(raw) == _columns()
    with pytest.raises(RuntimeError, match="header SHA-256 changed"):
        parse_v661_lookup_header(raw.replace(b"int64", b"uint64", 1))


def test_cohort_manifest_is_deterministic_and_labels_v661_root() -> None:
    rows = [
        {
            **_candidate("2", (7, 8, 9), legacy_cave_id=22),
            "nucleus_id": 202,
            "v661_pt_root_id": 2000,
        },
        {
            **_candidate("1", (1, 2, 3), legacy_cave_id=11),
            "nucleus_id": 101,
            "v661_pt_root_id": 1000,
        },
    ]
    encoded = cohort_csv_bytes(rows)
    assert encoded == cohort_csv_bytes(list(reversed(rows)))
    assert b"legacy_cave_id" in encoded
    assert b"v661_pt_root_id" in encoded
    assert b"v117_pt_root_id" not in encoded


def test_v661_join_uses_corrected_lookup_position_not_original_centroid() -> None:
    candidates = [_candidate("7", (1, 2, 3), legacy_cave_id=999999)]
    raw_csv = b"101,100.0,864691135737000001,700,111,90,91,92,1,2,3\r\n"
    rows, report = validate_v661_lookup_table(
        gzip.compress(raw_csv, mtime=0),
        columns=_columns(),
        candidate_rows=candidates,
    )
    assert report["validation_ok"] is True
    assert report["missing_candidate_points"] == 0
    assert report["join_to"] == [
        "pt_position_lookup_x",
        "pt_position_lookup_y",
        "pt_position_lookup_z",
    ]
    assert rows[0]["nucleus_id"] == 101
    assert rows[0]["v661_pt_root_id"] == 864691135737000001


def test_v661_lookup_hard_stops_on_missing_nonunique_or_zero_root() -> None:
    candidate = [_candidate("7", (1, 2, 3))]

    missing_csv = b"101,100.0,1,1,111,9,9,9,4,5,6\r\n"
    _, missing = validate_v661_lookup_table(
        gzip.compress(missing_csv, mtime=0),
        columns=_columns(),
        candidate_rows=candidate,
    )
    assert missing["validation_ok"] is False
    assert missing["missing_candidate_points"] == 1

    duplicate_csv = (
        b"101,100.0,1,1,111,9,9,9,1,2,3\r\n"
        b"102,101.0,2,2,112,8,8,8,1,2,3\r\n"
    )
    _, duplicate = validate_v661_lookup_table(
        gzip.compress(duplicate_csv, mtime=0),
        columns=_columns(),
        candidate_rows=candidate,
    )
    assert duplicate["validation_ok"] is False
    assert duplicate["nonunique_candidate_points"] == 1

    zero_root_csv = b"101,100.0,0,1,111,9,9,9,1,2,3\r\n"
    _, zero_root = validate_v661_lookup_table(
        gzip.compress(zero_root_csv, mtime=0),
        columns=_columns(),
        candidate_rows=candidate,
    )
    assert zero_root["exact_lookup_join_ok"] is True
    assert zero_root["root_integrity_ok"] is False
    assert zero_root["nonpositive_v661_root_points"] == 1
    assert zero_root["validation_ok"] is False


def test_v661_join_excludes_all_roi_rows_for_duplicate_canonical_nucleus() -> None:
    candidates = [_candidate("1", (1, 2, 3)), _candidate("2", (4, 5, 6))]
    raw_csv = (
        b"202,100.0,22,1,111,9,9,9,1,2,3\r\n"
        b"202,101.0,33,2,112,8,8,8,4,5,6\r\n"
    )
    rows, report = validate_v661_lookup_table(
        gzip.compress(raw_csv, mtime=0),
        columns=_columns(),
        candidate_rows=candidates,
    )
    assert rows == []
    assert report["exact_lookup_join_ok"] is True
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
    assert coreg["canonical_identity"] == "nucleus_detection_lookup_v1.id"
    assert coreg["coreg_source"] == COREG_SOURCE
    assert coreg["nwb_pt_root_id_exact_integer_safe"] is False
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

    with pytest.raises(RuntimeError, match="partial point"):
        scan_hdf5_metadata(make_h5(1.0, np.nan, 3.0), asset_path="asset")
    with pytest.raises(RuntimeError, match="not exact non-negative integer"):
        scan_hdf5_metadata(make_h5(1.5, 2.0, 3.0), asset_path="asset")
