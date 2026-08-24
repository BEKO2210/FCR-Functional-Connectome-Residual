"""Experiment 006 Stage B1: bounded H01 synapse-export schema probe.

The probe intentionally reports structure only: no neuron IDs, synapse IDs,
counts, degrees, densities, or model outcomes are emitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fcr.data.h01_edges import probe_schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = {
        "experiment": "006-stage-b1-schema-probe",
        **probe_schema(args.input),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
