"""Experiment 006 Stage B1: extract one deterministic shard partition."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from fcr.data.h01_edges import (
    FROZEN_NODE_SHA256,
    H01_SHARD_BYTES,
    H01_SHARD_COUNT,
    assigned_shards,
    download_shard,
    extract_selected_pairs,
    load_frozen_node_ids,
    shard_name,
    write_pair_counts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes-csv", required=True)
    parser.add_argument("--part-index", required=True, type=int)
    parser.add_argument("--part-count", required=True, type=int)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    node_ids = load_frozen_node_ids(args.nodes_csv)
    selected_ids = set(node_ids)
    shard_indices = assigned_shards(args.part_index, args.part_count)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    combined: Counter[tuple[int, int]] = Counter()
    shard_reports: list[dict[str, object]] = []

    for index in shard_indices:
        shard_path = work_dir / shard_name(index)
        download_report = download_shard(index, shard_path)
        try:
            counts, extraction_report = extract_selected_pairs(shard_path, selected_ids)
        finally:
            shard_path.unlink(missing_ok=True)
        combined.update(counts)
        shard_reports.append({**download_report, **extraction_report})

    write_pair_counts(combined, args.counts)
    report = {
        "experiment": "006-stage-b1-extract-part",
        "part_index": args.part_index,
        "part_count": args.part_count,
        "frozen_node_sha256": FROZEN_NODE_SHA256,
        "frozen_node_count": len(node_ids),
        "source_shard_count": H01_SHARD_COUNT,
        "assigned_shards": shard_indices,
        "assigned_source_bytes": sum(H01_SHARD_BYTES[index] for index in shard_indices),
        "shards": shard_reports,
        "partial_selected_pairs_including_autapses": len(combined),
        "partial_selected_synapses": int(sum(combined.values())),
        "model_metrics_computed": False,
    }

    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
