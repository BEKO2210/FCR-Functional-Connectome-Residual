"""Reusable experiment runner for structural predictive coding baselines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .metrics import binary_code_length_bits, bits_per_positive, bits_per_sample
from .models import DistanceBinnedModel, GlobalBernoulliModel, TypeDistanceModel, TypePairModel
from .schema import ConnectomeSample


@dataclass(frozen=True)
class ModelResult:
    model: str
    residual_bits: float
    model_bits: int
    two_part_bits: float
    bits_per_pair: float
    bits_per_positive: float


def evaluate_models(train: ConnectomeSample, test: ConnectomeSample) -> list[ModelResult]:
    models: list[tuple[str, Any]] = [
        ("global", GlobalBernoulliModel()),
        ("distance", DistanceBinnedModel()),
        ("type-pair", TypePairModel()),
        ("type+distance", TypeDistanceModel()),
    ]
    results: list[ModelResult] = []
    for name, model in models:
        model.fit(train)
        p = model.predict_proba(test)
        residual = binary_code_length_bits(test.connected, p)
        overhead = int(model.model_bits())
        total = residual + overhead
        results.append(
            ModelResult(
                model=name,
                residual_bits=residual,
                model_bits=overhead,
                two_part_bits=total,
                bits_per_pair=bits_per_sample(total, len(test)),
                bits_per_positive=bits_per_positive(total, test.connected),
            )
        )
    return results


def save_results(results: list[ModelResult], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([asdict(r) for r in results], indent=2) + "\n",
        encoding="utf-8",
    )
