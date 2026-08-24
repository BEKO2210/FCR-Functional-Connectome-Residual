from __future__ import annotations

import numpy as np

from fcr.advanced_models import GeometryLogisticModel, geometry_features
from fcr.metrics import binary_code_length_bits
from fcr.models import GlobalBernoulliModel
from fcr.schema import ConnectomeSample


def _geometry_sample(n_nodes: int = 20) -> ConnectomeSample:
    node_ids = np.arange(n_nodes, dtype=np.int64)
    xyz = np.column_stack(
        [
            np.arange(n_nodes, dtype=float) * 10.0,
            (np.arange(n_nodes) % 4).astype(float) * 5.0,
            (np.arange(n_nodes) % 3).astype(float) * 7.0,
        ]
    )
    source = np.repeat(node_ids, n_nodes)
    target = np.tile(node_ids, n_nodes)
    keep = source != target
    source = source[keep]
    target = target[keep]
    source_xyz = xyz[source]
    target_xyz = xyz[target]
    distance = np.linalg.norm(source_xyz - target_xyz, axis=1)
    connected = (distance < 42.0).astype(np.int8)
    return ConnectomeSample(
        source=source,
        target=target,
        source_type=np.full(len(source), "e"),
        target_type=np.full(len(source), "e"),
        distance=distance,
        connected=connected,
        source_xyz=source_xyz,
        target_xyz=target_xyz,
    )


def test_geometry_survives_subset() -> None:
    sample = _geometry_sample()
    mask = np.arange(len(sample)) % 2 == 0
    subset = sample.subset(mask)
    assert subset.source_xyz is not None
    assert subset.target_xyz is not None
    assert subset.source_xyz.shape == (int(mask.sum()), 3)
    assert np.allclose(subset.source_xyz, sample.source_xyz[mask])


def test_distance_logistic_beats_global_on_distance_signal() -> None:
    sample = _geometry_sample()
    split = len(sample) // 2
    train = sample.subset(np.arange(len(sample)) < split)
    test = sample.subset(np.arange(len(sample)) >= split)

    global_model = GlobalBernoulliModel().fit(train)
    geometry_model = GeometryLogisticModel(feature_set="distance", l2=1.0).fit(train)
    global_bits = binary_code_length_bits(test.connected, global_model.predict_proba(test))
    geometry_bits = binary_code_length_bits(test.connected, geometry_model.predict_proba(test))

    assert geometry_model.converged_
    assert geometry_model.iterations_ <= geometry_model.max_iter
    assert geometry_bits < global_bits
    assert geometry_model.model_bits() == 64 * 4  # intercept+coef, mean, scale


def test_geometry_feature_dimensions_are_frozen() -> None:
    sample = _geometry_sample(8)
    assert geometry_features(sample, "distance").shape[1] == 1
    assert geometry_features(sample, "relative").shape[1] == 7
    assert geometry_features(sample, "spatial").shape[1] == 16
