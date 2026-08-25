"""Experiment 007 Stage B: MICrONS-only solver equivalence replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from fcr.data.microns_public_l23 import load_public_l23_candidate_data
from fcr.experiment_007 import compare_microns_equivalence, run_nested_spatial_cv_damped
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
        raise RuntimeError("Experiment 007 MICrONS replay requires pair geometry")
    if len(data.node_ids) != 334:
        raise RuntimeError(f"unexpected MICrONS node count: {len(data.node_ids)}")
    if int(np.asarray(data.sample.connected).sum()) <= 0:
        raise RuntimeError("MICrONS graph contains no observed connections")

    original = run_nested_spatial_cv(
        data.sample,
        data.node_ids,
        data.coordinates_nm,
    )
    repaired = run_nested_spatial_cv_damped(
        data.sample,
        data.node_ids,
        data.coordinates_nm,
    )
    equivalence = compare_microns_equivalence(original, repaired)

    report = {
        "experiment": "007",
        "stage": "B-microns-solver-equivalence",
        "preregistration_issue": 22,
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
        "original_evaluation": original,
        "repaired_evaluation": repaired,
        "equivalence": equivalence,
        "h01_accessed": False,
        "interpretation_guardrail": (
            "Experiment 007 Stage B is numerical development on MICrONS only. "
            "It does not produce an H01 biological transfer result."
        ),
    }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
