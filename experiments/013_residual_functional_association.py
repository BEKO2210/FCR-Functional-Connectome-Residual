from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from fcr.experiment_013 import run_experiment_013


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functional-archive", required=True)
    parser.add_argument("--soma", required=True)
    parser.add_argument("--synapses", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--edges", required=True)
    args = parser.parse_args()

    report, edge_csv = run_experiment_013(
        functional_archive=args.functional_archive,
        soma_csv=args.soma,
        synapse_csv=args.synapses,
    )
    report["runtime"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(args.edges).write_bytes(edge_csv)
    print(json.dumps(report["primary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
