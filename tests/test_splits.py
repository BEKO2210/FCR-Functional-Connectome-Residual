import numpy as np

from fcr.schema import ConnectomeSample
from fcr.splits import contiguous_axis_node_split, spatial_sample_split


def test_contiguous_split_has_disjoint_nodes():
    node_ids = np.arange(20)
    xyz = np.column_stack([node_ids, np.zeros(20), np.zeros(20)])
    split = contiguous_axis_node_split(node_ids, xyz)

    assert set(split.train).isdisjoint(split.validation)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.validation).isdisjoint(split.test)
    assert len(split.train) + len(split.validation) + len(split.test) == 20


def test_spatial_pair_sets_share_no_nodes():
    node_ids = np.arange(12)
    xyz = np.column_stack([node_ids, np.zeros(12), np.zeros(12)])
    split = contiguous_axis_node_split(
        node_ids,
        xyz,
        train_fraction=0.5,
        validation_fraction=0.25,
    )

    source = np.repeat(node_ids, len(node_ids))
    target = np.tile(node_ids, len(node_ids))
    keep = source != target
    source = source[keep]
    target = target[keep]
    n = len(source)
    sample = ConnectomeSample(
        source=source,
        target=target,
        source_type=np.full(n, "x"),
        target_type=np.full(n, "x"),
        distance=np.abs(source - target).astype(float),
        connected=((source + target) % 7 == 0).astype(int),
    )

    train, validation, test = spatial_sample_split(sample, split)
    train_nodes = set(train.source) | set(train.target)
    validation_nodes = set(validation.source) | set(validation.target)
    test_nodes = set(test.source) | set(test.target)

    assert train_nodes.isdisjoint(validation_nodes)
    assert train_nodes.isdisjoint(test_nodes)
    assert validation_nodes.isdisjoint(test_nodes)
