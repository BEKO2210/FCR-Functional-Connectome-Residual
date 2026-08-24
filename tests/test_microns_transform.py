import numpy as np
import pandas as pd
import pytest

from fcr.data.microns_transform import aggregate_synapses, build_candidate_data, normalize_nodes


def _nodes():
    return pd.DataFrame(
        {
            "pt_root_id": [30, 10, 20],
            "pt_position_x": [0.0, 0.0, 3.0],
            "pt_position_y": [4.0, 0.0, 0.0],
            "pt_position_z": [0.0, 0.0, 0.0],
            "cell_type": ["C", "A", "B"],
        }
    )


def test_normalize_nodes_sorts_and_preserves_nm_coordinates():
    nodes = normalize_nodes(_nodes())
    assert nodes["pt_root_id"].tolist() == [10, 20, 30]
    assert nodes.loc[1, "pt_position_x"] == 3.0


def test_aggregate_synapses_counts_multiplicity_and_removes_autapses():
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [10, 10, 10, 20],
            "post_pt_root_id": [20, 20, 10, 30],
        }
    )
    out = aggregate_synapses(synapses)
    counts = {
        (row.pre_pt_root_id, row.post_pt_root_id): row.synapse_count
        for row in out.itertuples(index=False)
    }
    assert counts == {(10, 20): 2, (20, 30): 1}


def test_build_candidate_data_has_all_directed_nonself_pairs():
    synapses = pd.DataFrame(
        {
            "pre_pt_root_id": [10, 10, 20],
            "post_pt_root_id": [20, 20, 30],
        }
    )
    data = build_candidate_data(_nodes(), synapses)
    sample = data.sample
    assert len(sample) == 6
    pairs = list(zip(sample.source.tolist(), sample.target.tolist(), strict=True))
    assert pairs == [(10, 20), (10, 30), (20, 10), (20, 30), (30, 10), (30, 20)]
    counts = dict(zip(pairs, data.synapse_count.tolist(), strict=True))
    assert counts[(10, 20)] == 2
    assert counts[(20, 30)] == 1
    assert counts[(10, 30)] == 0
    idx = pairs.index((10, 30))
    assert sample.distance[idx] == pytest.approx(4.0)
    assert np.array_equal(sample.connected, (data.synapse_count > 0).astype(np.int8))


def test_candidate_pair_guard_prevents_accidental_quadratic_explosion():
    with pytest.raises(ValueError, match="candidate graph would contain"):
        build_candidate_data(
            _nodes(),
            pd.DataFrame(columns=["pre_pt_root_id", "post_pt_root_id"]),
            max_candidate_pairs=5,
        )


def test_duplicate_root_ids_are_rejected():
    nodes = _nodes()
    nodes.loc[1, "pt_root_id"] = nodes.loc[0, "pt_root_id"]
    with pytest.raises(ValueError, match="must be unique"):
        normalize_nodes(nodes)
