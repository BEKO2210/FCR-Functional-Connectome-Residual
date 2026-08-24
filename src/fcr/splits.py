"""Leakage-resistant split helpers for connectome experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import ConnectomeSample


@dataclass(frozen=True)
class SpatialNodeSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def contiguous_axis_node_split(
    node_ids: np.ndarray,
    coordinates: np.ndarray,
    *,
    axis: int = 0,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> SpatialNodeSplit:
    """Split nodes into contiguous spatial slabs along one coordinate axis.

    This is intentionally stricter than a random edge split: a node belongs to
    exactly one partition, and primary within-partition pair sets share no nodes.
    """
    ids = np.asarray(node_ids)
    xyz = np.asarray(coordinates, dtype=float)
    if ids.ndim != 1:
        raise ValueError("node_ids must be one-dimensional")
    if xyz.ndim != 2 or xyz.shape[0] != len(ids) or xyz.shape[1] < 1:
        raise ValueError("coordinates must have shape (n_nodes, n_dimensions)")
    if not 0 <= axis < xyz.shape[1]:
        raise ValueError("axis is out of bounds")
    if len(ids) < 3:
        raise ValueError("at least three nodes are required")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("node_ids must be unique")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train + validation fractions must be < 1")

    order = np.argsort(xyz[:, axis], kind="stable")
    n = len(ids)
    train_end = max(1, min(n - 2, int(round(n * train_fraction))))
    validation_end = max(
        train_end + 1,
        min(n - 1, int(round(n * (train_fraction + validation_fraction)))),
    )
    ordered_ids = ids[order]
    return SpatialNodeSplit(
        train=ordered_ids[:train_end],
        validation=ordered_ids[train_end:validation_end],
        test=ordered_ids[validation_end:],
    )


def within_node_set_mask(sample: ConnectomeSample, node_ids: np.ndarray) -> np.ndarray:
    """Select candidate pairs whose source and target both belong to node_ids."""
    allowed = np.asarray(node_ids)
    return np.isin(sample.source, allowed) & np.isin(sample.target, allowed)


def spatial_sample_split(
    sample: ConnectomeSample,
    split: SpatialNodeSplit,
) -> tuple[ConnectomeSample, ConnectomeSample, ConnectomeSample]:
    """Return pair samples fully contained within each node partition."""
    return (
        sample.subset(within_node_set_mask(sample, split.train)),
        sample.subset(within_node_set_mask(sample, split.validation)),
        sample.subset(within_node_set_mask(sample, split.test)),
    )
