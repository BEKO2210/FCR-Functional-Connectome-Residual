from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fcr.data.h01_nodes import central_rank_window, eligible_h01_neurons


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "c3_rep_manual": [10, 20, 30, 30, 40, 50, 60, 70],
            "celltype": [
                "PYRAMIDAL",
                "INTERNEURON",
                "PYRAMIDAL",
                "ASTROCYTE",
                "OLIGO",
                "SPINY_STELLATE",
                "UNKNOWN",
                "UNCLASSIFIED_NEURON",
            ],
            "layer": ["L2", "L3", "L4", "L4", "L5", "L6", "L1", "L2"],
            "x": [4, 1, 2, 3, 5, 6, 7, 8],
            "y": [1, 2, 3, 4, 5, 6, 7, 8],
            "z": [2, 3, 4, 5, 6, 7, 8, 9],
        }
    )


def test_non_neuronal_soma_with_same_c3_does_not_disqualify_neuron() -> None:
    result = eligible_h01_neurons(_frame())

    # C3 id 30 has one neuronal soma and one astrocyte soma. The reconciled
    # published rule counts neuron-labeled somas, so the neuron remains eligible.
    assert result["c3_rep_manual"].tolist() == [20, 30, 10, 50, 70]
    assert set(result["celltype"]) == {
        "PYRAMIDAL",
        "INTERNEURON",
        "SPINY_STELLATE",
        "UNCLASSIFIED_NEURON",
    }


def test_multiple_neuron_labeled_somas_for_same_c3_are_excluded() -> None:
    frame = _frame()
    duplicate_neuron = pd.DataFrame(
        {
            "c3_rep_manual": [20],
            "celltype": ["PYRAMIDAL"],
            "layer": ["L3"],
            "x": [9],
            "y": [9],
            "z": [9],
        }
    )
    frame = pd.concat([frame, duplicate_neuron], ignore_index=True)

    result = eligible_h01_neurons(frame)
    assert 20 not in set(result["c3_rep_manual"])


def test_coordinate_conversion_uses_frozen_h01_voxel_scale() -> None:
    result = eligible_h01_neurons(_frame())
    row = result.loc[result["c3_rep_manual"] == 20].iloc[0]

    assert row["x_nm"] == 8
    assert row["y_nm"] == 16
    assert row["z_nm"] == 99


def test_central_rank_window_is_deterministic_and_contiguous() -> None:
    eligible = pd.DataFrame(
        {
            "c3_rep_manual": np.arange(100, 110, dtype=np.int64),
            "x": np.arange(10, dtype=np.int64),
        }
    )
    selected = central_rank_window(eligible, count=4)
    assert selected["c3_rep_manual"].tolist() == [103, 104, 105, 106]


def test_rejects_missing_required_columns() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        eligible_h01_neurons(pd.DataFrame({"c3_rep_manual": [1]}))


def test_rejects_non_integer_c3_identity() -> None:
    frame = _frame()
    frame["c3_rep_manual"] = frame["c3_rep_manual"].astype(float)
    frame.loc[0, "c3_rep_manual"] = 10.5
    with pytest.raises(ValueError, match="non-integer"):
        eligible_h01_neurons(frame)
