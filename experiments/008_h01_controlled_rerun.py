"""Experiment 008: controlled H01 rerun with the validated Experiment 007 solver."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from fcr.data.h01_edges import FROZEN_NODE_COUNT, FROZEN_NODE_SHA256, sha256_file
from fcr.data.h01_experiment_006 import (
    FROZEN_CANDIDATE_ROWS,
    FROZEN_CANDIDATE_SHA256,
    FROZEN_CONNECTED_PAIRS,
    FROZEN_SELECTED_SYNAPSES,
    FROZEN_SPARSE_EDGE_SHA256,
    load_frozen_h01_experiment_006,
)
from fcr.experiment_008 import run_controlled_h01_rerun

PREREGISTRATION_ISSUE = 24
STAGE_B1_RUN_ID = 32791426287
STAGE_B1_COMMIT = "3ee8c7af6a7c2cf0b447afd7d91aaa67f72f3dc0"
STAGE_B1_ARTIFACT_ID = 9543477507
EXPERIMENT_007_MAIN = "33ffdd0f957d5c17d8047b6e4cba4b92a492154c"
EXPERIMENT_007_RUN_ID = 32797220405
EXPERIMENT_007_ARTIFACT_ID = 9545252968


def _validate_stage_b1_audit(path: str | Path) -> dict[str, object]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "stage_b1_extraction_complete": True,
        "frozen_node_sha256": FROZEN_NODE_SHA256,
        "frozen_node_count": FROZEN_NODE_COUNT,
        "source_shards": 166,
        "source_bytes": 126_066_253_429,
        "candidate_rows": FROZEN_CANDIDATE_ROWS,
        "connected_nonself_pairs": FROZEN_CONNECTED_PAIRS,
        "selected_nonself_synapses": FROZEN_SELECTED_SYNAPSES,
        "selected_autapse_synapses_excluded_from_primary": 0,
        "sparse_edges_sha256": FROZEN_SPARSE_EDGE_SHA256,
        "candidate_table_sha256": FROZEN_CANDIDATE_SHA256,
        "model_metrics_computed": False,
        "confirmation_decision_computed": False,
    }
    for key, expected_value in expected.items():
        if report.get(key) != expected_value:
            raise RuntimeError(
                f"Stage B1 audit mismatch for {key}: {report.get(key)!r} != {expected_value!r}"
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes-csv", required=True)
    parser.add_argument("--candidate-table", required=True)
    parser.add_argument("--sparse-edges", required=True)
    parser.add_argument("--stage-b1-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    stage_b1 = _validate_stage_b1_audit(args.stage_b1_audit)
    if sha256_file(args.sparse_edges) != FROZEN_SPARSE_EDGE_SHA256:
        raise RuntimeError("H01 Experiment 008 sparse-edge artifact hash mismatch")

    data = load_frozen_h01_experiment_006(args.nodes_csv, args.candidate_table)
    precheck, evaluation, strict_confirmation = run_controlled_h01_rerun(
        data.sample,
        data.node_ids,
        data.coordinates_nm,
    )

    report = {
        "experiment": "008",
        "stage": "controlled-h01-rerun",
        "preregistration_issue": PREREGISTRATION_ISSUE,
        "dataset": "H01 human temporal cortex c3 public release",
        "evidence_level": "E2-independent-cross-dataset-human-cortex-controlled-rerun",
        "stage_b1": {
            "run_id": STAGE_B1_RUN_ID,
            "commit": STAGE_B1_COMMIT,
            "artifact_id": STAGE_B1_ARTIFACT_ID,
            "audit": stage_b1,
        },
        "solver_repair": {
            "experiment": "007",
            "validated_main": EXPERIMENT_007_MAIN,
            "microns_equivalence_run": EXPERIMENT_007_RUN_ID,
            "artifact_id": EXPERIMENT_007_ARTIFACT_ID,
            "h01_accessed_during_solver_validation": False,
        },
        "input_sha256": {
            "primary_nodes.csv": sha256_file(args.nodes_csv),
            "h01_experiment_006_candidates.csv.gz": sha256_file(args.candidate_table),
            "h01_experiment_006_sparse_edges.csv.gz": sha256_file(args.sparse_edges),
        },
        "github": {
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "sha": os.environ.get("GITHUB_SHA"),
        },
        "graph": {
            "nodes": int(len(data.node_ids)),
            "directed_candidate_pairs": int(len(data.sample)),
            "connected_pairs": int(np.asarray(data.sample.connected).sum()),
            "observed_synapses": int(np.asarray(data.synapse_count).sum()),
        },
        "pre_model_stop_condition_check": precheck,
        "evaluation": evaluation,
        "independent_confirmation": strict_confirmation,
        "interpretation_guardrail": (
            "Experiment 008 is the controlled numerical completion of the preregistered H01 "
            "structural transfer test after Experiment 006 stopped before a score. Only the "
            "separately MICrONS-validated Experiment 007 solver is substituted. A positive "
            "result is structural cross-dataset evidence only; it does not establish functional "
            "equivalence, memory preservation, consciousness, identity, whole-brain emulation, "
            "or historical novelty."
        ),
    }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
