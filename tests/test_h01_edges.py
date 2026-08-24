from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path

from fcr.data import h01_edges


def _record(pre: object, post: object) -> dict[str, object]:
    return {
        "pre_synaptic_site": {"neuron_id": pre},
        "post_synaptic_partner": {"neuron_id": post},
        "confidence": 0.9,
    }


def test_frozen_shard_manifest_matches_stage_a_totals() -> None:
    assert h01_edges.H01_SHARD_COUNT == 166
    assert h01_edges.H01_TOTAL_BYTES == 126_066_253_429
    assert h01_edges.shard_name(0) == "export000000000000.json"
    assert h01_edges.shard_name(165) == "export000000000165.json"


def test_eight_parts_cover_each_shard_exactly_once() -> None:
    groups = [set(h01_edges.assigned_shards(index, 8)) for index in range(8)]
    union = set().union(*groups)
    assert union == set(range(166))
    for left_index, left in enumerate(groups):
        for right in groups[left_index + 1 :]:
            assert left.isdisjoint(right)


def test_extract_selected_pairs_keeps_all_selected_multiplicity(tmp_path: Path) -> None:
    records = [
        _record("1", "2"),
        _record("1", "2"),
        _record("2", "2"),
        _record("1", "9"),
        _record(None, "2"),
        {"pre_synaptic_site": {}, "post_synaptic_partner": {"neuron_id": "2"}},
    ]
    shard = tmp_path / "shard.json"
    shard.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    counts, report = h01_edges.extract_selected_pairs(shard, {1, 2, 3})

    assert counts == Counter({(1, 2): 2, (2, 2): 1})
    assert report["records_seen"] == 6
    assert report["missing_or_invalid_identity_records"] == 2
    assert report["selected_synapses"] == 3
    assert report["selected_autapse_synapses"] == 1


def test_pair_count_round_trip_and_deterministic_gzip(tmp_path: Path) -> None:
    counts = Counter({(1, 2): 3, (2, 2): 4, (3, 1): 1})
    partial = tmp_path / "part.csv"
    h01_edges.write_pair_counts(counts, partial)
    assert h01_edges.read_pair_counts([partial]) == counts

    first = tmp_path / "edges-a.csv.gz"
    second = tmp_path / "edges-b.csv.gz"
    first_report = h01_edges.write_sparse_edges_gzip(counts, first)
    second_report = h01_edges.write_sparse_edges_gzip(counts, second)

    assert first_report == second_report
    assert first_report["connected_nonself_pairs"] == 2
    assert first_report["selected_nonself_synapses"] == 4
    assert first_report["selected_autapse_synapses"] == 4
    assert h01_edges.sha256_file(first) == h01_edges.sha256_file(second)


def test_candidate_table_contains_every_ordered_nonself_pair_once(tmp_path: Path) -> None:
    counts = Counter({(1, 2): 2, (2, 2): 9, (3, 1): 1})
    destination = tmp_path / "candidate.csv.gz"
    report = h01_edges.write_candidate_table_gzip([1, 2, 3], counts, destination)

    with gzip.open(destination, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert report == {"candidate_rows": 6, "connected_nonself_pairs": 2}
    assert len(rows) == 6
    assert {(int(row["pre_id"]), int(row["post_id"])) for row in rows} == {
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 3),
        (3, 1),
        (3, 2),
    }
    lookup = {(int(row["pre_id"]), int(row["post_id"])): row for row in rows}
    assert lookup[(1, 2)]["synapse_count"] == "2"
    assert lookup[(1, 2)]["connected"] == "1"
    assert lookup[(2, 1)]["synapse_count"] == "0"
    assert lookup[(2, 1)]["connected"] == "0"
