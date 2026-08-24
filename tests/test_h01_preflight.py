from __future__ import annotations

from pathlib import Path

import pandas as pd

from fcr.data.h01_preflight import (
    choose_connectivity_source,
    inspect_soma_csv,
    summarize_object_metadata,
)


def test_soma_preflight_reports_schema_without_rows(tmp_path: Path) -> None:
    path = tmp_path / "somas.csv"
    pd.DataFrame(
        {
            "neuron_id": [10, 20, 20],
            "soma_x_nm": [1.0, 2.0, 3.0],
            "soma_y_nm": [4.0, 5.0, 6.0],
            "soma_z_nm": [7.0, 8.0, 9.0],
            "cell_type": ["neuron", "neuron", "glia"],
        }
    ).to_csv(path, index=False)

    report = inspect_soma_csv(path)
    assert report["row_count"] == 3
    assert report["coordinate_candidates"]["soma_x_nm"]["min"] == 1.0
    assert report["coordinate_candidates"]["soma_x_nm"]["max"] == 3.0
    assert report["label_candidates"]["cell_type"]["value_counts"] == {
        "glia": 1,
        "neuron": 2,
    }
    assert report["id_candidates"]["neuron_id"]["duplicated_id_count"] == 1
    assert report["id_candidates"]["neuron_id"]["rows_in_duplicated_ids"] == 2
    assert "rows" not in report
    assert "edges" not in report


def test_object_summary_uses_names_and_sizes_only() -> None:
    items = [
        {"name": "syn/json/a.json", "size": "100", "md5Hash": "secret-a"},
        {"name": "syn/json/b.json", "size": "250", "md5Hash": "secret-b"},
        {"name": "other/file", "size": "999"},
    ]
    report = summarize_object_metadata(items, prefix="syn/json/")
    assert report["object_count"] == 2
    assert report["total_bytes"] == 350
    assert report["min_object_bytes"] == 100
    assert report["max_object_bytes"] == 250
    assert report["objects"] == [
        {"name": "syn/json/a.json", "bytes": 100},
        {"name": "syn/json/b.json", "bytes": 250},
    ]
    assert "md5Hash" not in str(report)


def test_storage_choice_is_frozen_at_two_gibibytes() -> None:
    gib = 1024**3
    assert choose_connectivity_source(2 * gib) == "official-crest-sqlite"
    assert choose_connectivity_source(2 * gib + 1) == "official-h01-sharded-json"
    assert choose_connectivity_source(None) == "official-h01-sharded-json"
    assert choose_connectivity_source(0) == "official-h01-sharded-json"
