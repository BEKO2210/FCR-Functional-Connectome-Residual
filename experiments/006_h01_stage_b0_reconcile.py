"""Metadata-only reconciliation for the H01 15,730 single-soma neuron count.

Temporary diagnostic: no synapse or edge source is referenced or opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

NEURON_TYPES = {
    "PYRAMIDAL",
    "INTERNEURON",
    "SPINY_ATYPICAL",
    "SPINY_STELLATE",
    "UNCLASSIFIED_NEURON",
}


def _identity_stats(frame: pd.DataFrame, column: str, neuron_mask: pd.Series) -> dict[str, int]:
    values = pd.to_numeric(frame[column], errors="coerce")
    finite = values.notna() & np.isfinite(values)
    neuron_values = values[finite & neuron_mask]

    neuron_counts = neuron_values.value_counts()
    all_counts = values[finite].value_counts()
    neuron_exactly_once_mask = neuron_values.map(neuron_counts) == 1
    global_exactly_once_mask = neuron_values.map(all_counts) == 1

    return {
        "neuron_rows_with_finite_id": int(len(neuron_values)),
        "unique_ids_among_neuron_rows": int(neuron_values.nunique()),
        "neuron_rows_whose_id_occurs_once_among_neuron_rows": int(neuron_exactly_once_mask.sum()),
        "unique_ids_occurring_once_among_neuron_rows": int((neuron_counts == 1).sum()),
        "neuron_rows_whose_id_occurs_once_in_full_table": int(global_exactly_once_mask.sum()),
        "unique_ids_occurring_once_in_full_table_and_neuronal": int(
            neuron_values[global_exactly_once_mask].nunique()
        ),
        "extra_neuron_rows_beyond_unique_ids": int(len(neuron_values) - neuron_values.nunique()),
        "neuron_ids_with_multiple_neuron_rows": int((neuron_counts > 1).sum()),
        "neuron_rows_in_multi_neuron_id_groups": int(neuron_counts[neuron_counts > 1].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soma", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.soma)
    celltype = frame["celltype"].fillna("").astype(str).str.strip().str.upper()
    neuron_mask = celltype.isin(NEURON_TYPES)

    result = {
        "outcome_blind": True,
        "edge_content_accessed": False,
        "total_soma_rows": int(len(frame)),
        "neuron_label_rows": int(neuron_mask.sum()),
        "neuron_type_counts": {
            str(k): int(v)
            for k, v in celltype[neuron_mask].value_counts().sort_index().items()
        },
        "c3_rep_manual": _identity_stats(frame, "c3_rep_manual", neuron_mask),
        "c3_rep_strict": _identity_stats(frame, "c3_rep_strict", neuron_mask),
        "c2_rep_manual": _identity_stats(frame, "c2_rep_manual", neuron_mask),
        "c2_rep_strict": _identity_stats(frame, "c2_rep_strict", neuron_mask),
        "target_count": 15_730,
    }

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
