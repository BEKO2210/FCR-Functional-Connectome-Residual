"""Nested node-disjoint spatial evaluation frozen for Experiment 005."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, log2

import numpy as np

from .advanced_models import GeometryLogisticModel
from .metrics import binary_code_length_bits
from .models import GlobalBernoulliModel
from .schema import ConnectomeSample
from .splits import within_node_set_mask


@dataclass(frozen=True)
class PredictiveFoldResult:
    model: str
    l2: float | None
    residual_bits: float
    model_bits: int
    selection_bits: int
    two_part_bits: float
    iterations: int | None
    converged: bool


@dataclass(frozen=True)
class DegreeOracleResult:
    residual_bits: float
    degree_side_bits: int
    total_bits: float


def contiguous_node_slabs(
    node_ids: np.ndarray,
    coordinates: np.ndarray,
    *,
    n_splits: int = 5,
    axis: int = 0,
) -> list[np.ndarray]:
    ids = np.asarray(node_ids)
    xyz = np.asarray(coordinates, dtype=float)
    if ids.ndim != 1 or xyz.shape != (len(ids), 3):
        raise ValueError("node_ids and coordinates must have shapes (n,) and (n, 3)")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("node_ids must be unique")
    if n_splits < 3 or n_splits > len(ids):
        raise ValueError("n_splits must be between 3 and the number of nodes")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    order = np.argsort(xyz[:, axis], kind="stable")
    return [np.asarray(part) for part in np.array_split(ids[order], n_splits)]


def _sample_for_nodes(sample: ConnectomeSample, node_ids: np.ndarray) -> ConnectomeSample:
    return sample.subset(within_node_set_mask(sample, node_ids))


def _positive_count(sample: ConnectomeSample) -> int:
    return int(np.asarray(sample.connected, dtype=np.int8).sum())


def _fit_geometry(
    feature_set: str,
    l2: float,
    train: ConnectomeSample,
    test: ConnectomeSample,
    *,
    selection_bits: int,
) -> PredictiveFoldResult:
    model = GeometryLogisticModel(feature_set=feature_set, l2=l2).fit(train)
    probabilities = model.predict_proba(test)
    residual = binary_code_length_bits(test.connected, probabilities)
    model_bits = model.model_bits()
    return PredictiveFoldResult(
        model=feature_set,
        l2=float(l2),
        residual_bits=residual,
        model_bits=model_bits,
        selection_bits=selection_bits,
        two_part_bits=residual + model_bits + selection_bits,
        iterations=model.iterations_,
        converged=model.converged_,
    )


def _select_l2(
    sample: ConnectomeSample,
    slabs: list[np.ndarray],
    available_indices: list[int],
    feature_set: str,
    candidates: tuple[float, ...],
) -> tuple[float, list[dict[str, object]]]:
    scores: list[dict[str, object]] = []
    for l2 in candidates:
        residual_sum = 0.0
        inner_folds: list[dict[str, object]] = []
        for validation_index in available_indices:
            train_nodes = np.concatenate(
                [slabs[i] for i in available_indices if i != validation_index]
            )
            validation_nodes = slabs[validation_index]
            train = _sample_for_nodes(sample, train_nodes)
            validation = _sample_for_nodes(sample, validation_nodes)
            model = GeometryLogisticModel(feature_set=feature_set, l2=l2).fit(train)
            residual = binary_code_length_bits(
                validation.connected, model.predict_proba(validation)
            )
            residual_sum += residual
            inner_folds.append(
                {
                    "validation_slab": validation_index,
                    "train_pairs": len(train),
                    "train_positive_pairs": _positive_count(train),
                    "validation_pairs": len(validation),
                    "validation_positive_pairs": _positive_count(validation),
                    "residual_bits": residual,
                    "iterations": model.iterations_,
                }
            )
        scores.append(
            {
                "l2": float(l2),
                "residual_bits": residual_sum,
                "inner_folds": inner_folds,
            }
        )

    # Frozen tie-breaker from Issue #11: larger L2 is the simpler candidate.
    best = min(scores, key=lambda item: (float(item["residual_bits"]), -float(item["l2"])))
    return float(best["l2"]), scores


def degree_oracle_diagnostic(test: ConnectomeSample) -> DegreeOracleResult:
    """Directed degree-product codec with held-out degree vectors charged explicitly."""
    source = np.asarray(test.source)
    target = np.asarray(test.target)
    y = np.asarray(test.connected, dtype=np.int8)
    nodes = np.unique(np.concatenate([source, target]))
    n_nodes = len(nodes)
    if n_nodes < 2:
        raise ValueError("degree diagnostic requires at least two nodes")

    out_degree = {int(node): int(y[source == node].sum()) for node in nodes}
    in_degree = {int(node): int(y[target == node].sum()) for node in nodes}
    weights = np.asarray(
        [
            (out_degree[int(a)] + 0.5) * (in_degree[int(b)] + 0.5)
            for a, b in zip(source, target, strict=True)
        ],
        dtype=float,
    )
    target_edges = float(y.sum())
    if target_edges <= 0:
        probabilities = np.full(len(y), 0.5 / (len(y) + 1.0), dtype=float)
    else:
        low, high = 0.0, 1.0
        while float(np.minimum(high * weights, 1.0 - 1e-12).sum()) < target_edges:
            high *= 2.0
        for _ in range(80):
            middle = (low + high) / 2.0
            expected = float(np.minimum(middle * weights, 1.0 - 1e-12).sum())
            if expected < target_edges:
                low = middle
            else:
                high = middle
        probabilities = np.minimum(high * weights, 1.0 - 1e-12)

    residual = binary_code_length_bits(y, probabilities)
    width = max(1, ceil(log2(n_nodes)))
    side_bits = int(2 * n_nodes * width)
    return DegreeOracleResult(
        residual_bits=residual,
        degree_side_bits=side_bits,
        total_bits=residual + side_bits,
    )


def run_nested_spatial_cv(
    sample: ConnectomeSample,
    node_ids: np.ndarray,
    coordinates: np.ndarray,
) -> dict[str, object]:
    """Run exactly the Experiment 005 protocol from GitHub Issue #11."""
    slabs = contiguous_node_slabs(node_ids, coordinates, n_splits=5, axis=0)
    families: dict[str, tuple[float, ...]] = {
        "distance": (0.1, 1.0, 10.0),
        "relative": (0.1, 1.0, 10.0),
        "spatial": (1.0, 10.0, 100.0),
    }
    selection_bits = {name: ceil(log2(len(values))) for name, values in families.items()}
    outer_results: list[dict[str, object]] = []

    for outer_index in range(5):
        test_nodes = slabs[outer_index]
        available = [index for index in range(5) if index != outer_index]
        train_nodes = np.concatenate([slabs[index] for index in available])
        train = _sample_for_nodes(sample, train_nodes)
        test = _sample_for_nodes(sample, test_nodes)

        global_model = GlobalBernoulliModel().fit(train)
        global_residual = binary_code_length_bits(
            test.connected, global_model.predict_proba(test)
        )
        predictive: dict[str, dict[str, object]] = {
            "global": asdict(
                PredictiveFoldResult(
                    model="global",
                    l2=None,
                    residual_bits=global_residual,
                    model_bits=global_model.model_bits(),
                    selection_bits=0,
                    two_part_bits=global_residual + global_model.model_bits(),
                    iterations=None,
                    converged=True,
                )
            )
        }
        inner_selection: dict[str, object] = {}
        for family, candidates in families.items():
            selected_l2, scores = _select_l2(
                sample, slabs, available, family, candidates
            )
            inner_selection[family] = {
                "selected_l2": selected_l2,
                "candidates": scores,
            }
            predictive[family] = asdict(
                _fit_geometry(
                    family,
                    selected_l2,
                    train,
                    test,
                    selection_bits=selection_bits[family],
                )
            )

        outer_results.append(
            {
                "outer_slab": outer_index,
                "train_node_count": len(train_nodes),
                "test_node_count": len(test_nodes),
                "test_node_ids": [int(value) for value in test_nodes],
                "train_pairs": len(train),
                "train_positive_pairs": _positive_count(train),
                "test_pairs": len(test),
                "test_positive_pairs": _positive_count(test),
                "inner_selection": inner_selection,
                "predictive": predictive,
                "degree_oracle": asdict(degree_oracle_diagnostic(test)),
            }
        )

    model_names = ["global", "distance", "relative", "spatial"]
    aggregate: dict[str, dict[str, object]] = {}
    global_total = sum(
        float(fold["predictive"]["global"]["residual_bits"]) for fold in outer_results
    )
    for name in model_names:
        residual = sum(
            float(fold["predictive"][name]["residual_bits"]) for fold in outer_results
        )
        two_part = sum(
            float(fold["predictive"][name]["two_part_bits"]) for fold in outer_results
        )
        wins = sum(
            float(fold["predictive"][name]["residual_bits"])
            < float(fold["predictive"]["global"]["residual_bits"])
            for fold in outer_results
        )
        aggregate[name] = {
            "residual_bits": residual,
            "two_part_bits": two_part,
            "residual_reduction_vs_global": (
                0.0 if name == "global" else (global_total - residual) / global_total
            ),
            "wins_vs_global": int(wins),
        }

    eligible = ["distance", "relative", "spatial"]
    best_name = min(eligible, key=lambda name: float(aggregate[name]["residual_bits"]))
    reduction = float(aggregate[best_name]["residual_reduction_vs_global"])
    wins = int(aggregate[best_name]["wins_vs_global"])
    return {
        "protocol": "experiment-005-nested-spatial-cv",
        "evidence_level": "E1-development-nested-spatial-cv",
        "outer_axis": 0,
        "outer_slabs": 5,
        "folds": outer_results,
        "aggregate": aggregate,
        "primary": {
            "best_non_global_family": best_name,
            "residual_reduction_vs_global": reduction,
            "wins_vs_global": wins,
            "threshold_reduction": 0.05,
            "threshold_wins": 4,
            "criterion_met": bool(reduction >= 0.05 and wins >= 4),
        },
    }
