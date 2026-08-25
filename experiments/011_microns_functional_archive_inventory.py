from __future__ import annotations

import argparse
import json
from pathlib import Path

from fcr.data.microns_functional_archive import inventory_functional_archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory the frozen MICrONS functional archive")
    parser.add_argument("--archive", required=True, help="Path to the pinned functional archive")
    parser.add_argument("--output", required=True, help="Path for deterministic JSON report")
    args = parser.parse_args()

    archive_path = Path(args.archive)
    report = inventory_functional_archive(archive_path.read_bytes())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "Experiment 011 inventory complete: "
        f"classification={report['next_step_classification']}"
    )


if __name__ == "__main__":
    main()
