"""Small in-memory schemas for prototype experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConnectomeSample:
    """Candidate neuron pairs plus observed binary connectivity.

    ``source_xyz`` and ``target_xyz`` are optional observable side information.
    They are carried through subsets so geometry-aware codecs can be evaluated
    without changing the legacy structural baselines.
    """

    source: np.ndarray
    target: np.ndarray
    source_type: np.ndarray
    target_type: np.ndarray
    distance: np.ndarray
    connected: np.ndarray
    source_xyz: np.ndarray | None = None
    target_xyz: np.ndarray | None = None

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
        n = len(np.asarray(self.connected))
        if np.any(np.asarray(self.distance) < 0):
            raise ValueError("distance must be non-negative")
        y = np.asarray(self.connected)
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("connected must contain only 0 and 1")

        if (self.source_xyz is None) != (self.target_xyz is None):
            raise ValueError("source_xyz and target_xyz must either both be set or both be None")
        if self.source_xyz is not None and self.target_xyz is not None:
            source_xyz = np.asarray(self.source_xyz, dtype=float)
            target_xyz = np.asarray(self.target_xyz, dtype=float)
            if source_xyz.shape != (n, 3) or target_xyz.shape != (n, 3):
                raise ValueError("pair geometry must have shape (n_pairs, 3)")
            if not np.all(np.isfinite(source_xyz)) or not np.all(np.isfinite(target_xyz)):
                raise ValueError("pair geometry must contain only finite values")

    def subset(self, mask: np.ndarray) -> ConnectomeSample:
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
            source_xyz=(
                np.asarray(self.source_xyz)[mask] if self.source_xyz is not None else None
            ),
            target_xyz=(
                np.asarray(self.target_xyz)[mask] if self.target_xyz is not None else None
            ),
        )

    def __len__(self) -> int:
        return len(self.connected)
