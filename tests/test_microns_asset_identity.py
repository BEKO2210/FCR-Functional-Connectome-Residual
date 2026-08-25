from __future__ import annotations

import numpy as np
import pytest

from fcr.data.microns_asset_identity import (
    parse_frozen_asset_session_scan,
    scan_nwb_roi_identity_from_asset_path,
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


def _h5(*, session_id: FakeDataset | None = None) -> tuple[dict[str, object], FakeDataset]:
    plane_id = FakeDataset(np.array([5, 9], dtype=np.int64))
    h5: dict[str, object] = {
        "processing": {
            "ophys": {
                "ImageSegmentation": {
                    "PlaneSegmentation2": {"id": plane_id},
                }
            }
        }
    }
    if session_id is not None:
        h5["session_id"] = session_id
    return h5, plane_id


def test_frozen_asset_path_parses_preregistered_session_scan() -> None:
    assert parse_frozen_asset_session_scan() == (9, 4)

    with pytest.raises(RuntimeError, match="contradicts Experiment 010"):
        parse_frozen_asset_session_scan(
            "sub-17797/sub-17797_ses-8-scan-4_behavior+image+ophys.nwb"
        )

    with pytest.raises(RuntimeError, match="exactly one session/scan identity"):
        parse_frozen_asset_session_scan("sub-17797/no-session-here.nwb")


def test_missing_nwb_session_id_uses_asset_path_without_fake_value_read() -> None:
    h5, plane_id = _h5()
    report, rows = scan_nwb_roi_identity_from_asset_path(h5)

    assert report["session"] == 9
    assert report["scan_idx"] == 4
    assert report["session_identity_source"] == "frozen-dandi-asset-path"
    assert report["session_id_present_in_nwb"] is False
    assert report["session_id_diagnostic"] is None
    assert "/session_id" not in report["value_read_paths"]
    assert report["functional_values_read"] is False
    assert {(row["field"], row["mask_id"]) for row in rows} == {(2, 5), (2, 9)}
    assert plane_id.reads == 1


def test_present_nwb_session_id_is_consistency_diagnostic_only() -> None:
    session_id = FakeDataset(np.array(b"9-scan-4"))
    h5, plane_id = _h5(session_id=session_id)
    report, rows = scan_nwb_roi_identity_from_asset_path(h5)

    assert report["session_identity_source"] == "frozen-dandi-asset-path"
    assert report["session_id_present_in_nwb"] is True
    assert report["session_id_diagnostic"] == "9-scan-4"
    assert report["value_read_paths"][0] == "/session_id"
    assert len(rows) == 2
    assert session_id.reads == 1
    assert plane_id.reads == 1


def test_present_nwb_session_id_must_agree_with_frozen_path() -> None:
    session_id = FakeDataset(np.array(b"8-scan-4"))
    h5, plane_id = _h5(session_id=session_id)

    with pytest.raises(RuntimeError, match="contradicts frozen"):
        scan_nwb_roi_identity_from_asset_path(h5)
    assert session_id.reads == 1
    assert plane_id.reads == 0
