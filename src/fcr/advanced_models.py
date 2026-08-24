"""Parameter-efficient geometry-aware codecs for Experiment 005."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import ConnectomeSample


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def geometry_features(sample: ConnectomeSample, feature_set: str) -> np.ndarray:
    if sample.source_xyz is None or sample.target_xyz is None:
        raise ValueError("geometry-aware models require source_xyz and target_xyz")

    source_xyz = np.asarray(sample.source_xyz, dtype=float)
    target_xyz = np.asarray(sample.target_xyz, dtype=float)
    delta = target_xyz - source_xyz
    distance = np.asarray(sample.distance, dtype=float)
    safe_distance = np.maximum(distance, 1e-12)
    log_distance = np.log1p(distance)[:, None]

    if feature_set == "distance":
        return log_distance

    relative = np.column_stack(
        [
            log_distance[:, 0],
            np.abs(delta[:, 0]),
            np.abs(delta[:, 1]),
            np.abs(delta[:, 2]),
            delta[:, 0] / safe_distance,
            delta[:, 1] / safe_distance,
            delta[:, 2] / safe_distance,
        ]
    )
    if feature_set == "relative":
        return relative
    if feature_set == "spatial":
        midpoint = (source_xyz + target_xyz) / 2.0
        return np.column_stack([relative, source_xyz, target_xyz, midpoint])
    raise ValueError(f"unknown geometry feature set: {feature_set}")


@dataclass
class GeometryLogisticModel:
    """Deterministic ridge-logistic model with training-only standardization."""

    feature_set: str
    l2: float
    max_iter: int = 100
    tolerance: float = 1e-8
    coefficients_: np.ndarray | None = None
    means_: np.ndarray | None = None
    scales_: np.ndarray | None = None
    converged_: bool = False
    iterations_: int = 0

    def fit(self, sample: ConnectomeSample) -> GeometryLogisticModel:
        if self.l2 <= 0:
            raise ValueError("l2 must be positive")
        if len(sample) == 0:
            raise ValueError("cannot fit an empty sample")

        features = geometry_features(sample, self.feature_set)
        targets = np.asarray(sample.connected, dtype=float)
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

        converged = False
        iterations = 0
        for iteration in range(1, self.max_iter + 1):
            iterations = iteration
            probabilities = _sigmoid(design @ coefficients)
            weights = np.clip(probabilities * (1.0 - probabilities), 1e-8, None)
            gradient = design.T @ (targets - probabilities) - self.l2 * penalty * coefficients
            hessian = design.T @ (weights[:, None] * design) + self.l2 * np.diag(penalty)
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            coefficients += step
            if float(np.linalg.norm(step)) < self.tolerance:
                converged = True
                break

        self.coefficients_ = coefficients
        self.means_ = means
        self.scales_ = scales
        self.converged_ = converged
        self.iterations_ = iterations
        if not converged:
            raise RuntimeError(
                f"geometry logistic optimizer did not converge within {self.max_iter} iterations"
            )
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
        # Conservative two-part accounting frozen in Experiment 005: every fitted
        # coefficient and every training-derived mean/scale costs one float64.
        n_float_values = len(self.coefficients_) + len(self.means_) + len(self.scales_)
        return int(64 * n_float_values)
