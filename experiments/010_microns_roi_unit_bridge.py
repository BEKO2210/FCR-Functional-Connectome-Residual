"""Experiment 010 Stage A: provenance-native token-free MICrONS identity bridge."""

from __future__ import annotations

import argparse
import json

from fcr.data.microns_roi_unit_bridge_live import run_reconciled_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cohort-output", required=True)
    args = parser.parse_args()

    report = run_reconciled_preflight(args.output, args.cohort_output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
