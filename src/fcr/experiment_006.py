"""Strict independent-confirmation helpers for Experiment 006."""

from __future__ import annotations

import numpy as np

from .nested_cv import contiguous_node_slabs
from .schema import ConnectomeSample
from .splits import within_node_set_mask


def outer_slab_positive_precheck(
    sample: ConnectomeSample,
    node_ids: np.ndarray,
    coordinates_nm: np.ndarray,
) -> list[dict[str, int]]:
    """Validate the preregistered H01 stop condition before any model fitting."""
    slabs = contiguous_node_slabs(node_ids, coordinates_nm, n_splits=5, axis=0)
    results: list[dict[str, int]] = []
    for index, nodes in enumerate(slabs):
        mask = within_node_set_mask(sample, nodes)
        pairs = int(mask.sum())
        positives = int(np.asarray(sample.connected, dtype=np.int8)[mask].sum())
        if positives <= 0:
            raise RuntimeError(
                f"Experiment 006 stop condition: outer slab {index} has zero positive pairs"
            )
        results.append(
            {
                "outer_slab": index,
                "node_count": int(len(nodes)),
                "candidate_pairs": pairs,
                "positive_pairs": positives,
            }
        )
    return results


def strict_spatial_confirmation(evaluation: dict[str, object]) -> dict[str, object]:
    """Apply the frozen H01 criterion to spatial only, never best-of-family."""
    aggregate = evaluation.get("aggregate")
    if not isinstance(aggregate, dict):
        raise RuntimeError("Experiment 005 evaluator output is missing aggregate results")
    required = {"global", "distance", "relative", "spatial"}
    if not required.issubset(aggregate):
        raise RuntimeError("Experiment 005 evaluator output is missing a frozen model family")

    global_result = aggregate["global"]
    distance_result = aggregate["distance"]
    relative_result = aggregate["relative"]
    spatial_result = aggregate["spatial"]
    if not all(
        isinstance(value, dict)
        for value in (global_result, distance_result, relative_result, spatial_result)
    ):
        raise RuntimeError("Experiment 005 aggregate family result has an invalid type")

    reduction = float(spatial_result["residual_reduction_vs_global"])
    wins = int(spatial_result["wins_vs_global"])
    threshold_reduction = 0.05
    threshold_wins = 4
    criterion_met = reduction >= threshold_reduction and wins >= threshold_wins

    global_residual = float(global_result["residual_bits"])
    distance_residual = float(distance_result["residual_bits"])
    relative_residual = float(relative_result["residual_bits"])
    spatial_residual = float(spatial_result["residual_bits"])
    global_two_part = float(global_result["two_part_bits"])
    spatial_two_part = float(spatial_result["two_part_bits"])

    return {
        "primary_family": "spatial",
        "residual_reduction_vs_global": reduction,
        "wins_vs_global": wins,
        "threshold_reduction": threshold_reduction,
        "threshold_wins": threshold_wins,
        "criterion_met": bool(criterion_met),
        "spatial_residual_bits": spatial_residual,
        "global_residual_bits": global_residual,
        "spatial_two_part_bits": spatial_two_part,
        "global_two_part_bits": global_two_part,
        "spatial_two_part_better_than_global": bool(spatial_two_part < global_two_part),
        "secondary_replication": {
            "spatial_better_than_relative": bool(spatial_residual < relative_residual),
            "relative_better_or_equal_distance": bool(relative_residual <= distance_residual),
            "distance_better_than_global": bool(distance_residual < global_residual),
            "residual_bits": {
                "global": global_residual,
                "distance": distance_residual,
                "relative": relative_residual,
                "spatial": spatial_residual,
            },
        },
    }
