"""Experiment 005: stronger structural baselines with nested spatial CV."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from fcr.data.microns_public_l23 import load_public_l23_candidate_data
from fcr.nested_cv import run_nested_spatial_cv


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soma", required=True)
    parser.add_argument("--synapses", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = load_public_l23_candidate_data(args.soma, args.synapses)
    if data.sample.source_xyz is None or data.sample.target_xyz is None:
        raise RuntimeError("Experiment 005 requires pair geometry")
    if len(data.node_ids) < 100:
        raise RuntimeError("unexpectedly small public graph")
    if int(np.asarray(data.sample.connected).sum()) <= 0:
        raise RuntimeError("public graph contains no observed connections")

    evaluation = run_nested_spatial_cv(
        data.sample,
        data.node_ids,
        data.coordinates_nm,
    )
    report = {
        "experiment": "005",
        "preregistration_issue": 11,
        "dataset": "MICrONS layer-2/3 v185 public static release",
        "dataset_doi": "10.5281/zenodo.7510511",
        "input_sha256": {
            "soma_valence_v185.csv": _sha256(args.soma),
            "soma_subgraph_synapses_spines_v185.csv": _sha256(args.synapses),
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
        "evaluation": evaluation,
        "interpretation_guardrail": (
            "Experiment 005 is development evidence on a dataset already inspected in "
            "Experiment 004. Meeting the preregistered criterion is not independent "
            "confirmation and does not imply functional equivalence or whole-brain results."
        ),
    }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
