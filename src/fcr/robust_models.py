"""Numerically robust geometry-aware ridge-logistic model for Experiment 007.

This module does not replace the frozen Experiment 005 implementation. It
implements the separately preregistered Experiment 007 solver while reusing the
exact Experiment 005 geometry feature construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .advanced_models import geometry_features
from .schema import ConnectomeSample


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _penalized_objective(
    design: np.ndarray,
    targets: np.ndarray,
    coefficients: np.ndarray,
    l2: float,
    penalty: np.ndarray,
) -> float:
    logits = design @ coefficients
    data_term = np.logaddexp(0.0, logits).sum() - float(targets @ logits)
    penalty_term = 0.5 * l2 * float(np.dot(penalty * coefficients, coefficients))
    return float(data_term + penalty_term)


@dataclass
class DampedGeometryLogisticModel:
    """Deterministic Armijo-damped Newton ridge-logistic model.

    Numerical constants are frozen in GitHub Issue #22 before H01 evaluation.
    The statistical objective, features, standardization, initialization and
    model-bit accounting match Experiment 005.
    """

    feature_set: str
    l2: float
    max_iter: int = 500
    gradient_tolerance: float = 1e-10
    step_tolerance: float = 1e-8
    armijo_constant: float = 1e-4
    shrink_factor: float = 0.5
    max_backtracks: int = 31
    coefficients_: np.ndarray | None = None
    means_: np.ndarray | None = None
    scales_: np.ndarray | None = None
    converged_: bool = False
    iterations_: int = 0
    objective_history_: list[float] = field(default_factory=list)
    scaled_gradient_inf_: float | None = None
    backtracks_: int = 0

    def fit(self, sample: ConnectomeSample) -> DampedGeometryLogisticModel:
        if self.l2 <= 0:
            raise ValueError("l2 must be positive")
        if len(sample) == 0:
            raise ValueError("cannot fit an empty sample")
        if not 0.0 < self.armijo_constant < 1.0:
            raise ValueError("armijo_constant must be in (0, 1)")
        if not 0.0 < self.shrink_factor < 1.0:
            raise ValueError("shrink_factor must be in (0, 1)")
        if self.max_iter <= 0 or self.max_backtracks <= 0:
            raise ValueError("iteration limits must be positive")

        features = geometry_features(sample, self.feature_set)
        targets = np.asarray(sample.connected, dtype=float)
        if not np.all(np.isfinite(features)):
            raise ValueError("geometry features contain non-finite values")
        if not np.all((targets == 0.0) | (targets == 1.0)):
            raise ValueError("targets must be binary")

        means = features.mean(axis=0)
        scales = features.std(axis=0)
        scales = np.where(scales > 1e-12, scales, 1.0)
        standardized = (features - means) / scales
        design = np.column_stack([np.ones(len(standardized)), standardized])

        prevalence = (float(targets.sum()) + 0.5) / (len(targets) + 1.0)
        coefficients = np.zeros(design.shape[1], dtype=float)
        coefficients[0] = np.log(prevalence / (1.0 - prevalence))
        penalty = np.ones(len(coefficients), dtype=float)
        penalty[0] = 0.0

        objective = _penalized_objective(design, targets, coefficients, self.l2, penalty)
        if not np.isfinite(objective):
            raise RuntimeError("initial penalized logistic objective is non-finite")
        objective_history = [objective]
        converged = False
        iterations = 0
        total_backtracks = 0
        scaled_gradient_inf = float("inf")

        for iteration in range(1, self.max_iter + 1):
            iterations = iteration
            probabilities = _sigmoid(design @ coefficients)
            weights = np.clip(probabilities * (1.0 - probabilities), 1e-8, None)
            gradient = design.T @ (probabilities - targets) + self.l2 * penalty * coefficients
            scaled_gradient_inf = float(np.max(np.abs(gradient))) / max(1, len(targets))
            if scaled_gradient_inf <= self.gradient_tolerance:
                converged = True
                break

            hessian = design.T @ (weights[:, None] * design) + self.l2 * np.diag(penalty)
            try:
                direction = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                direction = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            if not np.all(np.isfinite(direction)):
                raise RuntimeError("damped Newton direction is non-finite")

            descent_measure = float(gradient @ direction)
            if not np.isfinite(descent_measure) or descent_measure <= 0.0:
                raise RuntimeError("damped Newton direction is not a descent direction")

            alpha = 1.0
            accepted = False
            candidate = coefficients
            candidate_objective = objective
            used_backtracks = 0
            for backtrack in range(self.max_backtracks):
                used_backtracks = backtrack
                candidate = coefficients - alpha * direction
                candidate_objective = _penalized_objective(
                    design,
                    targets,
                    candidate,
                    self.l2,
                    penalty,
                )
                armijo_bound = objective - self.armijo_constant * alpha * descent_measure
                if np.isfinite(candidate_objective) and candidate_objective <= armijo_bound:
                    accepted = True
                    break
                alpha *= self.shrink_factor

            total_backtracks += used_backtracks
            if not accepted:
                raise RuntimeError(
                    f"damped Newton line search failed after {self.max_backtracks} trials"
                )

            step = alpha * direction
            coefficients = candidate
            objective = candidate_objective
            objective_history.append(objective)
            if float(np.linalg.norm(step)) <= self.step_tolerance * (
                1.0 + float(np.linalg.norm(coefficients))
            ):
                converged = True
                break

        self.coefficients_ = coefficients
        self.means_ = means
        self.scales_ = scales
        self.converged_ = converged
        self.iterations_ = iterations
        self.objective_history_ = objective_history
        self.scaled_gradient_inf_ = scaled_gradient_inf
        self.backtracks_ = total_backtracks
        if not converged:
            message = (
                "damped geometry logistic optimizer did not converge within "
                f"{self.max_iter} iterations"
            )
            raise RuntimeError(message)
        return self

    def predict_proba(self, sample: ConnectomeSample) -> np.ndarray:
        if self.coefficients_ is None or self.means_ is None or self.scales_ is None:
            raise RuntimeError("model is not fitted")
        features = geometry_features(sample, self.feature_set)
        standardized = (features - self.means_) / self.scales_
        design = np.column_stack([np.ones(len(standardized)), standardized])
        return _sigmoid(design @ self.coefficients_)

    def model_bits(self) -> int:
        if self.coefficients_ is None or self.means_ is None or self.scales_ is None:
            raise RuntimeError("model is not fitted")
        n_float_values = len(self.coefficients_) + len(self.means_) + len(self.scales_)
        return int(64 * n_float_values)
