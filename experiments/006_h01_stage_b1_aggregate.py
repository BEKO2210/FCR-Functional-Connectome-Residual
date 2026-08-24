"""Experiment 006 Stage B1: combine all shard partitions into frozen edge artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fcr.data.h01_edges import (
    EXPECTED_CANDIDATE_PAIRS,
    FROZEN_NODE_COUNT,
    FROZEN_NODE_SHA256,
    H01_SHARD_COUNT,
    H01_TOTAL_BYTES,
    load_frozen_node_ids,
    read_pair_counts,
    sha256_file,
    write_candidate_table_gzip,
    write_sparse_edges_gzip,
)


def _load_part_reports(parts_dir: Path, part_count: int) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for index in range(part_count):
        path = parts_dir / f"part_{index}.json"
        if not path.exists():
            raise RuntimeError(f"missing Stage B1 part report: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["part_index"] != index or report["part_count"] != part_count:
            raise RuntimeError(f"partition metadata mismatch in {path}")
        if report["frozen_node_sha256"] != FROZEN_NODE_SHA256:
            raise RuntimeError(f"frozen node hash mismatch in {path}")
        reports.append(report)
    return reports


def _validate_shard_coverage(reports: list[dict[str, object]]) -> int:
    observed_indices: list[int] = []
    observed_source_bytes = 0
    for report in reports:
        observed_indices.extend(int(value) for value in report["assigned_shards"])
        observed_source_bytes += int(report["assigned_source_bytes"])
        for shard in report["shards"]:
            if shard["expected_bytes"] != shard["observed_bytes"]:
                raise RuntimeError(f"downloaded shard size mismatch: {shard['shard_index']}")

    if sorted(observed_indices) != list(range(H01_SHARD_COUNT)):
        raise RuntimeError("Stage B1 shard coverage is incomplete or duplicated")
    if observed_source_bytes != H01_TOTAL_BYTES:
        raise RuntimeError(
            f"Stage B1 source-byte mismatch: {observed_source_bytes} != {H01_TOTAL_BYTES}"
        )
    return observed_source_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes-csv", required=True)
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--part-count", required=True, type=int)
    parser.add_argument("--sparse-edges", required=True)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    node_ids = load_frozen_node_ids(args.nodes_csv)
    if len(node_ids) != FROZEN_NODE_COUNT:
        raise RuntimeError("Stage B1 aggregator received the wrong frozen node count")

    parts_dir = Path(args.parts_dir)
    reports = _load_part_reports(parts_dir, args.part_count)
    observed_source_bytes = _validate_shard_coverage(reports)

    count_paths = [parts_dir / f"part_{index}.csv" for index in range(args.part_count)]
    if not all(path.exists() for path in count_paths):
        raise RuntimeError("one or more Stage B1 partial pair-count files are missing")
    counts = read_pair_counts(count_paths)
    selected_set = set(node_ids)
    if any(pre not in selected_set or post not in selected_set for pre, post in counts):
        raise RuntimeError("Stage B1 partial edge counts contain a non-frozen node")

    sparse_report = write_sparse_edges_gzip(counts, args.sparse_edges)
    candidate_report = write_candidate_table_gzip(node_ids, counts, args.candidate_table)
    if candidate_report["candidate_rows"] != EXPECTED_CANDIDATE_PAIRS:
        raise RuntimeError("Stage B1 candidate graph does not contain every ordered non-self pair")
    if candidate_report["connected_nonself_pairs"] != sparse_report["connected_nonself_pairs"]:
        raise RuntimeError("sparse edge count disagrees with candidate graph positives")

    report = {
        "experiment": "006-stage-b1-complete-extraction",
        "stage_b1_extraction_complete": True,
        "frozen_node_sha256": FROZEN_NODE_SHA256,
        "frozen_node_count": len(node_ids),
        "source_shards": H01_SHARD_COUNT,
        "source_bytes": observed_source_bytes,
        "candidate_rows": candidate_report["candidate_rows"],
        "connected_nonself_pairs": candidate_report["connected_nonself_pairs"],
        "selected_nonself_synapses": sparse_report["selected_nonself_synapses"],
        "selected_autapse_synapses_excluded_from_primary": sparse_report[
            "selected_autapse_synapses"
        ],
        "sparse_edges_sha256": sha256_file(args.sparse_edges),
        "candidate_table_sha256": sha256_file(args.candidate_table),
        "model_metrics_computed": False,
        "confirmation_decision_computed": False,
    }

    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
