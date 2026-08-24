from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fcr.data.microns_public_l23 import build_public_l23_candidate_data


def _soma() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pt_root_id": [10, 20, 30, 40],
            "cell_type": ["e", "i", "e", "g"],
            "soma_x_nm": [0.0, 3.0, 10.0, 20.0],
            "soma_y_nm": [0.0, 4.0, 0.0, 0.0],
            "soma_z_nm": [0.0, 0.0, 0.0, 0.0],
        }
    )


def test_builds_complete_directed_nonself_candidate_graph() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [10, 10, 20, 30],
            "post_pt_root_id": [20, 20, 10, 40],
        }
    )
    data = build_public_l23_candidate_data(_soma(), synapses)

    assert data.node_ids.tolist() == [10, 20, 30, 40]
    assert len(data.sample) == 12
    assert int(data.sample.connected.sum()) == 3
    assert int(data.synapse_count.sum()) == 4
    assert not np.any(data.sample.source == data.sample.target)

    pair_to_count = {
        (int(source), int(target)): int(count)
        for source, target, count in zip(
            data.sample.source, data.sample.target, data.synapse_count, strict=True
        )
    }
    assert pair_to_count[(10, 20)] == 2
    assert pair_to_count[(20, 10)] == 1
    assert pair_to_count[(30, 40)] == 1
    assert pair_to_count[(40, 30)] == 0


def test_normalises_types_and_uses_soma_distance() -> None:
    synapses = pd.DataFrame(
        {
            "pre_root_id": [10, 20, 30],
            "post_root_id": [20, 30, 40],
        }
    )
    data = build_public_l23_candidate_data(_soma(), synapses)

    mask = (data.sample.source == 10) & (data.sample.target == 20)
    index = int(np.flatnonzero(mask)[0])
    assert data.sample.source_type[index] == "excitatory"
    assert data.sample.target_type[index] == "inhibitory"
    assert data.sample.distance[index] == pytest.approx(5.0)


def test_rejects_ambiguous_endpoint_column_detection() -> None:
    synapses = pd.DataFrame(
        {
            "pre_root_alpha_id": [10],
            "pre_root_beta_id": [10],
            "post_root_id": [20],
        }
    )
    with pytest.raises(ValueError, match="uniquely identify pre"):
        build_public_l23_candidate_data(_soma(), synapses)


def test_max_nodes_applies_after_endpoint_filtering() -> None:
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [20, 30, 40],
            "post_pt_root_id": [30, 40, 20],
        }
    )
    data = build_public_l23_candidate_data(_soma(), synapses, max_nodes=3)
    assert data.node_ids.tolist() == [20, 30, 40]
    assert len(data.sample) == 6
