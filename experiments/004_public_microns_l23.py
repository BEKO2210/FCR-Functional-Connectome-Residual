"""E1 structural pilot on the token-free MICrONS layer-2/3 v185 release."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fcr.data.microns_public_l23 import load_public_l23_candidate_data
from fcr.experiment import evaluate_models
from fcr.splits import contiguous_axis_node_split, spatial_sample_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soma", required=True)
    parser.add_argument("--synapses", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = load_public_l23_candidate_data(args.soma, args.synapses)
    split = contiguous_axis_node_split(
        data.node_ids,
        data.coordinates_nm,
        axis=0,
        train_fraction=0.70,
        validation_fraction=0.15,
    )
    train, validation, test = spatial_sample_split(data.sample, split)

    if len(train) == 0 or len(validation) == 0 or len(test) == 0:
        raise RuntimeError("spatial split produced an empty pair partition")
    if int(np.asarray(train.connected).sum()) == 0:
        raise RuntimeError("training partition contains no observed positive connections")

    validation_results = evaluate_models(train, validation)
    test_results = evaluate_models(train, test)

    report = {
        "evidence_level": "E1-structural-public-static-pilot",
        "dataset": "MICrONS layer-2/3 v185 public static release",
        "dataset_doi": "10.5281/zenodo.7510511",
        "selection": (
            "all unique soma-table roots appearing in the proofread soma-subgraph synapse file "
            "with finite coordinates"
        ),
        "split": {
            "method": "contiguous_axis_node_split",
            "axis": 0,
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "node_overlap_allowed": False,
            "train_nodes": int(len(split.train)),
            "validation_nodes": int(len(split.validation)),
            "test_nodes": int(len(split.test)),
        },
        "graph": {
            "nodes": int(len(data.node_ids)),
            "directed_candidate_pairs": int(len(data.sample)),
            "connected_pairs": int(np.asarray(data.sample.connected).sum()),
            "observed_synapses": int(np.asarray(data.synapse_count).sum()),
        },
        "partitions": {
            "train_pairs": int(len(train)),
            "train_positive_pairs": int(np.asarray(train.connected).sum()),
            "validation_pairs": int(len(validation)),
            "validation_positive_pairs": int(np.asarray(validation.connected).sum()),
            "test_pairs": int(len(test)),
            "test_positive_pairs": int(np.asarray(test.connected).sum()),
        },
        "validation_models": [asdict(item) for item in validation_results],
        "test_models": [asdict(item) for item in test_results],
        "interpretation_guardrail": (
            "This older static layer-2/3 dataset is a real-data structural pilot only. "
            "It does not establish current-v1822 performance, functional equivalence, "
            "whole-brain compressibility, or any claim about consciousness or identity."
        ),
    }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
