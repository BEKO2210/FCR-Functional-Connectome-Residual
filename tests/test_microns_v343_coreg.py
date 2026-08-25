from __future__ import annotations

import csv
import gzip
import io

import pytest

from fcr.data.microns_v343_coreg import (
    EXPECTED_FIELDS,
    parse_v343_coregistration,
    parse_v343_header,
)


def _header_bytes(columns: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for column in columns:
        writer.writerow([column, "synthetic_type"])
    return stream.getvalue().encode()


def _gzip_rows(columns: list[str], records: list[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for record in records:
        writer.writerow([record[column] for column in columns])
    return gzip.compress(stream.getvalue().encode())


def _record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": 1,
        "valid": "t",
        "pt_position_x": 342768,
        "pt_position_y": 110096,
        "pt_position_z": 16982,
        "pt_supervoxel_id": 111892900580825107,
        "pt_root_id": 864691135771728459,
        "session": 9,
        "scan_idx": 4,
        "unit_id": 5443,
    }
    record.update(overrides)
    return record


def test_header_sidecar_order_controls_data_mapping() -> None:
    columns = [
        "id",
        "valid",
        "pt_position_x",
        "pt_position_y",
        "pt_position_z",
        "session",
        "scan_idx",
        "unit_id",
        "pt_supervoxel_id",
        "pt_root_id",
    ]
    header = _header_bytes(columns)
    data = _gzip_rows(columns, [_record()])

    by_unit, report = parse_v343_coregistration(data, header)

    assert report["schema"] == columns
    assert report["source_rows"] == 1
    assert report["target_session_scan_rows"] == 1
    assert by_unit[5443] == {(864691135771728459, 342768, 110096, 16982)}


def test_non_target_rows_do_not_enter_mapping() -> None:
    columns = sorted(EXPECTED_FIELDS)
    header = _header_bytes(columns)
    data = _gzip_rows(columns, [_record(session=8), _record(id=2, unit_id=17)])

    by_unit, report = parse_v343_coregistration(data, header)

    assert report["source_rows"] == 2
    assert report["target_session_scan_rows"] == 1
    assert 5443 not in by_unit
    assert 17 in by_unit


def test_header_rejects_missing_field() -> None:
    columns = sorted(EXPECTED_FIELDS - {"pt_root_id"})
    with pytest.raises(RuntimeError, match="field set"):
        parse_v343_header(_header_bytes(columns))


def test_header_rejects_extra_field() -> None:
    columns = sorted(EXPECTED_FIELDS) + ["unexpected"]
    with pytest.raises(RuntimeError, match="field set"):
        parse_v343_header(_header_bytes(columns))


def test_header_rejects_duplicate_field() -> None:
    columns = sorted(EXPECTED_FIELDS)
    columns.append(columns[0])
    with pytest.raises(RuntimeError, match="duplicates column"):
        parse_v343_header(_header_bytes(columns))


def test_data_rejects_row_width_mismatch() -> None:
    columns = sorted(EXPECTED_FIELDS)
    header = _header_bytes(columns)
    data = gzip.compress(b"1,2,3\n")
    with pytest.raises(RuntimeError, match="row width mismatch"):
        parse_v343_coregistration(data, header)


def test_target_invalid_row_is_hard_stop() -> None:
    columns = sorted(EXPECTED_FIELDS)
    header = _header_bytes(columns)
    data = _gzip_rows(columns, [_record(valid="f")])
    with pytest.raises(RuntimeError, match="marked invalid"):
        parse_v343_coregistration(data, header)


def test_invalid_gzip_is_hard_stop() -> None:
    columns = sorted(EXPECTED_FIELDS)
    with pytest.raises(RuntimeError, match="not valid gzip"):
        parse_v343_coregistration(b"not-gzip", _header_bytes(columns))
