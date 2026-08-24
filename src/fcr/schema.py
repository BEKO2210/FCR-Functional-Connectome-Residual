"""Small in-memory schemas for prototype experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConnectomeSample:
    """Candidate neuron pairs plus observed binary connectivity."""

    source: np.ndarray
    target: np.ndarray
    source_type: np.ndarray
    target_type: np.ndarray
    distance: np.ndarray
    connected: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.source,
            self.target,
            self.source_type,
            self.target_type,
            self.distance,
            self.connected,
        )
        lengths = {len(np.asarray(a)) for a in arrays}
        if len(lengths) != 1:
            raise ValueError("all sample arrays must have equal length")
        if np.any(np.asarray(self.distance) < 0):
            raise ValueError("distance must be non-negative")
        y = np.asarray(self.connected)
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("connected must contain only 0 and 1")

    def subset(self, mask: np.ndarray) -> "ConnectomeSample":
        mask = np.asarray(mask, dtype=bool)
        if len(mask) != len(self.connected):
            raise ValueError("mask length must match sample")
        return ConnectomeSample(
            source=np.asarray(self.source)[mask],
            target=np.asarray(self.target)[mask],
            source_type=np.asarray(self.source_type)[mask],
            target_type=np.asarray(self.target_type)[mask],
            distance=np.asarray(self.distance)[mask],
            connected=np.asarray(self.connected)[mask],
        )

    def __len__(self) -> int:
        return len(self.connected)
