"""Controlled H01 rerun using only the validated Experiment 007 solver repair."""

from __future__ import annotations

import numpy as np

from .experiment_006 import outer_slab_positive_precheck, strict_spatial_confirmation
from .experiment_007 import run_nested_spatial_cv_damped
from .schema import ConnectomeSample


def run_controlled_h01_rerun(
    sample: ConnectomeSample,
    node_ids: np.ndarray,
    coordinates_nm: np.ndarray,
) -> tuple[list[dict[str, int]], dict[str, object], dict[str, object]]:
    """Run the frozen H01 protocol with only the Experiment 007 solver substituted."""
    precheck = outer_slab_positive_precheck(sample, node_ids, coordinates_nm)
    evaluation = run_nested_spatial_cv_damped(sample, node_ids, coordinates_nm)
    if evaluation.get("protocol") != "experiment-005-nested-spatial-cv":
        raise RuntimeError("Experiment 008 did not execute the frozen Experiment 005 protocol")
    confirmation = strict_spatial_confirmation(evaluation)
    return precheck, evaluation, confirmation
