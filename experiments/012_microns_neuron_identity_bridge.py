#!/usr/bin/env python3
"""Run preregistered Experiment 012 without opening functional measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fcr.data.microns_neuron_identity import (
    SOMA_MD5,
    SYNAPSE_MD5,
    run_identity_bridge,
)


def _md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--soma", required=True)
    parser.add_argument("--synapses", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    archive_path = Path(args.archive)
    soma_path = Path(args.soma)
    synapse_path = Path(args.synapses)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)

    soma_observed_md5 = _md5_file(soma_path)
    synapse_observed_md5 = _md5_file(synapse_path)
    if soma_observed_md5 != SOMA_MD5:
        raise RuntimeError("soma_valence_v185.csv MD5 mismatch")
    if synapse_observed_md5 != SYNAPSE_MD5:
        raise RuntimeError("soma_subgraph_synapses_spines_v185.csv MD5 mismatch")

    report, manifest = run_identity_bridge(
        functional_archive=archive_path,
        soma_csv=soma_path,
        synapse_csv=synapse_path,
    )
    structural_source = report["structural_source"]
    assert isinstance(structural_source, dict)
    structural_source["soma_observed_md5"] = soma_observed_md5
    structural_source["synapse_observed_md5"] = synapse_observed_md5

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
