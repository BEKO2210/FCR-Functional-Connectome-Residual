"""Experiment 007 solver-repair replay helpers."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from . import nested_cv
from .robust_models import DampedGeometryLogisticModel
from .schema import ConnectomeSample

FAMILIES = ("global", "distance", "relative", "spatial")
GEOMETRY_FAMILIES = ("distance", "relative", "spatial")
MAX_RESIDUAL_DIFFERENCE_BITS = 0.1


def run_nested_spatial_cv_damped(
    sample: ConnectomeSample,
    node_ids: np.ndarray,
    coordinates: np.ndarray,
) -> dict[str, object]:
    """Reuse the frozen Experiment 005 evaluator while replacing only its solver class."""
    with patch.object(
        nested_cv,
        "GeometryLogisticModel",
        DampedGeometryLogisticModel,
    ):
        return nested_cv.run_nested_spatial_cv(sample, node_ids, coordinates)


def compare_microns_equivalence(
    original: dict[str, object],
    repaired: dict[str, object],
    *,
    residual_tolerance_bits: float = MAX_RESIDUAL_DIFFERENCE_BITS,
) -> dict[str, object]:
    """Enforce the preregistered Experiment 007 MICrONS equivalence gate."""
    if original.get("protocol") != "experiment-005-nested-spatial-cv":
        raise RuntimeError("original replay is not the frozen Experiment 005 protocol")
    if repaired.get("protocol") != "experiment-005-nested-spatial-cv":
        raise RuntimeError("repaired replay is not the frozen Experiment 005 protocol")

    original_folds = original.get("folds")
    repaired_folds = repaired.get("folds")
    if not isinstance(original_folds, list) or not isinstance(repaired_folds, list):
        raise RuntimeError("nested-CV fold output is missing")
    if len(original_folds) != 5 or len(repaired_folds) != 5:
        raise RuntimeError("MICrONS equivalence requires exactly five outer folds")

    fold_checks: list[dict[str, object]] = []
    max_fold_residual_difference = 0.0
    for original_fold, repaired_fold in zip(original_folds, repaired_folds, strict=True):
        if original_fold["outer_slab"] != repaired_fold["outer_slab"]:
            raise RuntimeError("outer-slab identity changed in repaired replay")
        if original_fold["test_node_ids"] != repaired_fold["test_node_ids"]:
            raise RuntimeError("outer test nodes changed in repaired replay")
        if original_fold["test_pairs"] != repaired_fold["test_pairs"]:
            raise RuntimeError("outer candidate graph changed in repaired replay")

        family_checks: dict[str, object] = {}
        for family in FAMILIES:
            original_predictive = original_fold["predictive"][family]
            repaired_predictive = repaired_fold["predictive"][family]
            difference = abs(
                float(original_predictive["residual_bits"])
                - float(repaired_predictive["residual_bits"])
            )
            max_fold_residual_difference = max(max_fold_residual_difference, difference)
            if difference > residual_tolerance_bits:
                raise RuntimeError(
                    f"MICrONS fold residual mismatch for {family}: {difference} bits"
                )
            if original_predictive["model_bits"] != repaired_predictive["model_bits"]:
                raise RuntimeError(f"model-bit accounting changed for {family}")
            if family != "global" and repaired_predictive["converged"] is not True:
                raise RuntimeError(f"repaired MICrONS fit did not converge for {family}")
            family_checks[family] = {
                "residual_difference_bits": difference,
                "model_bits_equal": True,
                "repaired_converged": bool(repaired_predictive["converged"]),
            }

        for family in GEOMETRY_FAMILIES:
            old_l2 = original_fold["inner_selection"][family]["selected_l2"]
            new_l2 = repaired_fold["inner_selection"][family]["selected_l2"]
            if old_l2 != new_l2:
                raise RuntimeError(
                    f"MICrONS selected L2 changed for {family}: {old_l2} != {new_l2}"
                )

        fold_checks.append(
            {
                "outer_slab": original_fold["outer_slab"],
                "families": family_checks,
                "selected_l2_equal": True,
            }
        )

    original_aggregate = original.get("aggregate")
    repaired_aggregate = repaired.get("aggregate")
    if not isinstance(original_aggregate, dict) or not isinstance(repaired_aggregate, dict):
        raise RuntimeError("nested-CV aggregate output is missing")

    aggregate_checks: dict[str, object] = {}
    max_aggregate_residual_difference = 0.0
    for family in FAMILIES:
        old = original_aggregate[family]
        new = repaired_aggregate[family]
        difference = abs(float(old["residual_bits"]) - float(new["residual_bits"]))
        max_aggregate_residual_difference = max(max_aggregate_residual_difference, difference)
        if difference > residual_tolerance_bits:
            raise RuntimeError(
                f"MICrONS aggregate residual mismatch for {family}: {difference} bits"
            )
        if int(old["wins_vs_global"]) != int(new["wins_vs_global"]):
            raise RuntimeError(f"MICrONS outer-fold win count changed for {family}")
        aggregate_checks[family] = {
            "residual_difference_bits": difference,
            "wins_vs_global": int(new["wins_vs_global"]),
        }

    old_primary = original.get("primary")
    new_primary = repaired.get("primary")
    if not isinstance(old_primary, dict) or not isinstance(new_primary, dict):
        raise RuntimeError("nested-CV primary output is missing")
    if old_primary["best_non_global_family"] != new_primary["best_non_global_family"]:
        raise RuntimeError("MICrONS best non-global family changed")
    if bool(old_primary["criterion_met"]) != bool(new_primary["criterion_met"]):
        raise RuntimeError("MICrONS Experiment 005 criterion disposition changed")

    return {
        "criterion_met": True,
        "residual_tolerance_bits": residual_tolerance_bits,
        "max_fold_residual_difference_bits": max_fold_residual_difference,
        "max_aggregate_residual_difference_bits": max_aggregate_residual_difference,
        "folds": fold_checks,
        "aggregate": aggregate_checks,
        "best_non_global_family": new_primary["best_non_global_family"],
        "experiment_005_criterion_met": bool(new_primary["criterion_met"]),
    }
