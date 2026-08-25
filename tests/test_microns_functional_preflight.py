from __future__ import annotations

import numpy as np
import pytest

from fcr.data.microns_functional_preflight import (
    FROZEN_ASSET_PATH,
    _read_allowed_value,
    cohort_csv_bytes,
    scan_hdf5_metadata,
    summarize_root_ids,
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
        {"asset_path": "asset", "plane": "PlaneSegmentation2", "roi_id": "2", "pt_root_id": 20},
        {"asset_path": "asset", "plane": "PlaneSegmentation1", "roi_id": "1", "pt_root_id": 10},
    ]
    reverse = list(reversed(rows))

    assert cohort_csv_bytes(rows) == cohort_csv_bytes(reverse)


def test_scan_reads_coregistration_but_not_functional_values() -> None:
    root_ids = FakeDataset(np.array([0, 1001, 1002], dtype=np.uint64))
    roi_ids = FakeDataset(np.array([1, 2, 3], dtype=np.int64))
    image_mask = FakeDataset(np.zeros((3, 2, 2), dtype=np.float32))
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
                        "id": roi_ids,
                        "pt_root_id": root_ids,
                        "image_mask": image_mask,
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
        "intervals": {
            "Clip": {
                "id": clip_ids,
                "start_time": clip_start,
            }
        },
    }

    report, manifest = scan_hdf5_metadata(h5, asset_path=FROZEN_ASSET_PATH)

    assert report["functional_values_read"] is False
    assert report["coregistration"]["root_id_precision_safe"] is True
    assert report["coregistration"]["candidate_rows"] == 2
    assert manifest is not None
    assert b"1001" in manifest and b"1002" in manifest

    assert root_ids.reads == 1
    assert roi_ids.reads == 1
    assert image_mask.reads == 0
    assert fluorescence_data.reads == 0
    assert fluorescence_timestamps.reads == 0
    assert clip_start.reads == 0
    assert clip_ids.reads == 0
