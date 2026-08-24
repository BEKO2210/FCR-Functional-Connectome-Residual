"""Experiment 006 Stage B0: freeze H01 nodes from soma metadata only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fcr.data.h01_nodes import freeze_h01_nodes

EXPECTED_SOMA_SHA256 = "cc38c8670bbeb3d61f58af44a529392111a59196ca0b9e0f74849b0afe1ffaff"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soma", required=True)
    parser.add_argument("--selected-csv", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    report = freeze_h01_nodes(args.soma, args.selected_csv)
    if report["soma_sha256"] != EXPECTED_SOMA_SHA256:
        raise RuntimeError(
            "canonical H01 soma hash changed: "
            f"observed={report['soma_sha256']} expected={EXPECTED_SOMA_SHA256}"
        )

    payload = {
        "experiment": "006-stage-b0-node-freeze",
        "preregistration_issue": 13,
        "schema_mapping_comment": 5402766985,
        "coordinate_mapping_comment": 5402772650,
        "outcome_blind": True,
        "edge_content_accessed": False,
        **report,
        "guardrail": (
            "This node set was frozen from canonical H01 soma metadata only. No H01 synapse "
            "or edge content was opened, counted, filtered, or used for selection."
        ),
    }

    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
