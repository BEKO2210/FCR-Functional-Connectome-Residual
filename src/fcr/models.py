"""Transparent probabilistic baselines for structural coding."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schema import ConnectomeSample


def _beta_mean(successes: int, trials: int, alpha: float) -> float:
    return (successes + alpha) / (trials + 2.0 * alpha)


@dataclass
class GlobalBernoulliModel:
    alpha: float = 0.5
    probability_: float | None = None

    def fit(self, sample: ConnectomeSample) -> "GlobalBernoulliModel":
        y = np.asarray(sample.connected, dtype=np.int8)
        self.probability_ = _beta_mean(int(y.sum()), len(y), self.alpha)
        return self

    def predict_proba(self, sample: ConnectomeSample) -> np.ndarray:
        if self.probability_ is None:
            raise RuntimeError("model is not fitted")
        return np.full(len(sample), self.probability_, dtype=float)

    def model_bits(self) -> int:
        return 64


@dataclass
class DistanceBinnedModel:
    n_bins: int = 12
    alpha: float = 0.5
    edges_: np.ndarray | None = None
    probabilities_: np.ndarray | None = None
    fallback_: float | None = None

    def fit(self, sample: ConnectomeSample) -> "DistanceBinnedModel":
        d = np.asarray(sample.distance, dtype=float)
        y = np.asarray(sample.connected, dtype=np.int8)
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        if len(d) == 0:
            raise ValueError("cannot fit an empty sample")
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        edges = np.quantile(d, quantiles)
        edges = np.maximum.accumulate(edges)
        if edges[-1] == edges[0]:
            edges[-1] = edges[0] + 1.0
        self.edges_ = edges
        self.fallback_ = _beta_mean(int(y.sum()), len(y), self.alpha)
        idx = np.clip(np.searchsorted(edges[1:-1], d, side="right"), 0, self.n_bins - 1)
        probs = np.empty(self.n_bins, dtype=float)
        for b in range(self.n_bins):
            mask = idx == b
            probs[b] = (
                _beta_mean(int(y[mask].sum()), int(mask.sum()), self.alpha)
                if mask.any()
                else self.fallback_
            )
        self.probabilities_ = probs
        return self

    def predict_proba(self, sample: ConnectomeSample) -> np.ndarray:
        if self.edges_ is None or self.probabilities_ is None:
            raise RuntimeError("model is not fitted")
        d = np.asarray(sample.distance, dtype=float)
        idx = np.clip(np.searchsorted(self.edges_[1:-1], d, side="right"), 0, self.n_bins - 1)
        return self.probabilities_[idx]

    def model_bits(self) -> int:
        return int(64 * (2 * self.n_bins + 1))


@dataclass
class TypePairModel:
    alpha: float = 0.5
    probabilities_: dict[tuple[str, str], float] = field(default_factory=dict)
    fallback_: float | None = None

    def fit(self, sample: ConnectomeSample) -> "TypePairModel":
        y = np.asarray(sample.connected, dtype=np.int8)
        st = np.asarray(sample.source_type).astype(str)
        tt = np.asarray(sample.target_type).astype(str)
        self.fallback_ = _beta_mean(int(y.sum()), len(y), self.alpha)
        self.probabilities_.clear()
        for key in sorted(set(zip(st, tt, strict=True))):
            mask = (st == key[0]) & (tt == key[1])
            self.probabilities_[key] = _beta_mean(int(y[mask].sum()), int(mask.sum()), self.alpha)
        return self

    def predict_proba(self, sample: ConnectomeSample) -> np.ndarray:
        if self.fallback_ is None:
            raise RuntimeError("model is not fitted")
        st = np.asarray(sample.source_type).astype(str)
        tt = np.asarray(sample.target_type).astype(str)
        return np.asarray(
            [self.probabilities_.get((a, b), self.fallback_) for a, b in zip(st, tt, strict=True)],
            dtype=float,
        )

    def model_bits(self) -> int:
        return int(64 * (len(self.probabilities_) + 1))


@dataclass
class TypeDistanceModel:
    """Cell-type pair probabilities conditioned on distance bins."""

    n_bins: int = 12
    alpha: float = 0.5
    edges_: np.ndarray | None = None
    probabilities_: dict[tuple[str, str, int], float] = field(default_factory=dict)
    fallback_: float | None = None

    def fit(self, sample: ConnectomeSample) -> "TypeDistanceModel":
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        d = np.asarray(sample.distance, dtype=float)
        y = np.asarray(sample.connected, dtype=np.int8)
        st = np.asarray(sample.source_type).astype(str)
        tt = np.asarray(sample.target_type).astype(str)
        edges = np.quantile(d, np.linspace(0.0, 1.0, self.n_bins + 1))
        edges = np.maximum.accumulate(edges)
        if edges[-1] == edges[0]:
            edges[-1] = edges[0] + 1.0
        self.edges_ = edges
        self.fallback_ = _beta_mean(int(y.sum()), len(y), self.alpha)
        bins = np.clip(np.searchsorted(edges[1:-1], d, side="right"), 0, self.n_bins - 1)
        self.probabilities_.clear()
        keys = sorted(set(zip(st, tt, bins.tolist(), strict=True)))
        for a, b, k in keys:
            mask = (st == a) & (tt == b) & (bins == k)
            self.probabilities_[(a, b, int(k))] = _beta_mean(
                int(y[mask].sum()), int(mask.sum()), self.alpha
            )
        return self

    def predict_proba(self, sample: ConnectomeSample) -> np.ndarray:
        if self.edges_ is None or self.fallback_ is None:
            raise RuntimeError("model is not fitted")
        d = np.asarray(sample.distance, dtype=float)
        st = np.asarray(sample.source_type).astype(str)
        tt = np.asarray(sample.target_type).astype(str)
        bins = np.clip(
            np.searchsorted(self.edges_[1:-1], d, side="right"), 0, self.n_bins - 1
        )
        return np.asarray(
            [
                self.probabilities_.get((a, b, int(k)), self.fallback_)
                for a, b, k in zip(st, tt, bins, strict=True)
            ],
            dtype=float,
        )

    def model_bits(self) -> int:
        n_float_params = (
            len(self.probabilities_) + len(self.edges_) + 1
            if self.edges_ is not None
            else 0
        )
        return int(64 * n_float_params)
