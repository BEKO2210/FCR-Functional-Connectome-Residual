"""Preregistered MICrONS residual-to-function association test (Experiment 013)."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import pickle
import tarfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .advanced_models import GeometryLogisticModel
from .data.microns_public_l23 import PublicL23Data, load_public_l23_candidate_data
from .nested_cv import _sample_for_nodes, _select_l2, contiguous_node_slabs

EXPERIMENT = "013"
PREREGISTRATION_ISSUE = 35
EVIDENCE_LEVEL = "E1-functional-association-pilot"

FUNCTIONAL_ARCHIVE_URL = (
    "https://zenodo.org/record/6363348/files/211019_vignette_functional_analysis_data.tgz"
)
FUNCTIONAL_ARCHIVE_MD5 = "bcb0d4f678909fbd481ac0b01242ae5c"
FUNCTIONAL_ARCHIVE_SHA256 = "5ee95299b1955832f3f7b07defcea842f95bdd0bb351cb9828581f96b96d868b"
NESTED_MEMBER = "function_data_tables.tgz"
NESTED_MEMBER_SIZE = 79_195_487
NESTED_MEMBER_SHA256 = "be972c2468bb4c32cca508870da02082697d9f3a37ce649743c21ce6c5f7a5b4"
TUNING_MEMBER = "EASETuning.pkl"
TUNING_MEMBER_SIZE = 25_441
TUNING_MEMBER_SHA256 = "c45bb2ec024c1fc2699a34e82087ded0ed515ca5706d28d4ae4ff848e213a6ce"
ALLEN_REFERENCE_COMMIT = "3f53e35e2bfd3063469dcbab0e1dedce5a82e3ca"

SOMA_MD5 = "b7c349b1ecbdad46185bb98d88b96d20"
SYNAPSE_MD5 = "5bbb8ff59dcad4ccea6d46930920cd04"
EXPECTED_STRUCTURAL_NODES = 334
RNG_SEED = 13013
RANDOM_REPLICATES = 10_000
PRIMARY_MIN_IMPROVEMENT = 0.05
PRIMARY_MAX_P = 0.05

_UINT64_MAX = (1 << 64) - 1


class _RestrictedTuningUnpickler(pickle.Unpickler):
    _ALLOWED_GLOBALS = {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"blocked pickle global: {module}.{name}")
        return super().find_class(module, name)


@dataclass(frozen=True)
class TuningData:
    segment_id: np.ndarray
    scan_id: np.ndarray
    osi: np.ndarray
    osi_p: np.ndarray


@dataclass(frozen=True)
class EligibleEdge:
    fold: int
    source: int
    target: int
    distance_nm: float
    probability: float
    surprise_bits: float
    source_scan: int
    target_scan: int
    source_osi: float
    target_osi: float
    source_osi_p: float
    target_osi_p: float
    distance_bin: int = -1
    high: bool = False
    low: bool = False

    @property
    def same_scan(self) -> bool:
        return self.source_scan == self.target_scan

    @property
    def orientation_difference(self) -> float:
        return abs(self.source_osi - self.target_osi)


def _md5(raw: bytes) -> str:
    return hashlib.md5(raw, usedforsecurity=False).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_member(raw: bytes, name: str, *, label: str) -> tuple[bytes, int]:
    try:
        archive = tarfile.open(fileobj=io.BytesIO(raw), mode="r:*")
    except tarfile.TarError as exc:
        raise RuntimeError(f"{label} is not a valid tar archive") from exc
    with archive:
        matches = [member for member in archive.getmembers() if member.name == name]
        if len(matches) != 1:
            raise RuntimeError(f"{label} must contain exactly one {name!r}")
        member = matches[0]
        if not member.isfile():
            raise RuntimeError(f"{name!r} is not a regular file")
        handle = archive.extractfile(member)
        if handle is None:
            raise RuntimeError(f"could not read {name!r}")
        payload = handle.read()
        if len(payload) != member.size:
            raise RuntimeError(f"size mismatch while reading {name!r}")
        return payload, int(member.size)


def extract_verified_tuning_pickle(functional_archive_raw: bytes) -> tuple[bytes, dict[str, object]]:
    if _md5(functional_archive_raw) != FUNCTIONAL_ARCHIVE_MD5:
        raise RuntimeError("functional archive MD5 mismatch")
    if _sha256(functional_archive_raw) != FUNCTIONAL_ARCHIVE_SHA256:
        raise RuntimeError("functional archive SHA-256 mismatch")

    nested, nested_size = _read_member(
        functional_archive_raw, NESTED_MEMBER, label="functional archive"
    )
    if nested_size != NESTED_MEMBER_SIZE or _sha256(nested) != NESTED_MEMBER_SHA256:
        raise RuntimeError("nested functional archive integrity mismatch")

    tuning, tuning_size = _read_member(nested, TUNING_MEMBER, label="nested functional archive")
    if tuning_size != TUNING_MEMBER_SIZE or _sha256(tuning) != TUNING_MEMBER_SHA256:
        raise RuntimeError("EASETuning.pkl integrity mismatch")

    return tuning, {
        "archive_url": FUNCTIONAL_ARCHIVE_URL,
        "archive_md5": FUNCTIONAL_ARCHIVE_MD5,
        "archive_sha256": FUNCTIONAL_ARCHIVE_SHA256,
        "nested_member": NESTED_MEMBER,
        "nested_sha256": NESTED_MEMBER_SHA256,
        "tuning_member": TUNING_MEMBER,
        "tuning_size_bytes": tuning_size,
        "tuning_sha256": TUNING_MEMBER_SHA256,
    }


def _restricted_load(raw: bytes) -> Mapping[str, object]:
    try:
        value = _RestrictedTuningUnpickler(io.BytesIO(raw)).load()
    except (pickle.UnpicklingError, EOFError, ValueError, TypeError) as exc:
        raise RuntimeError("EASETuning.pkl failed restricted deserialization") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("EASETuning.pkl top-level object is not a mapping")
    return value


def _positive_uint64(values: object, label: str) -> np.ndarray:
    if isinstance(values, (str, bytes)) or not isinstance(values, (Sequence, np.ndarray)):
        raise RuntimeError(f"{label} is not a one-dimensional sequence")
    arr = np.asarray(values)
    if arr.ndim != 1 or np.issubdtype(arr.dtype, np.floating):
        raise RuntimeError(f"{label} must contain exact one-dimensional integers")
    normalized: list[int] = []
    for item in arr.tolist():
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise RuntimeError(f"{label} must contain exact integers")
        value = int(item)
        if value <= 0 or value > _UINT64_MAX:
            raise RuntimeError(f"{label} outside positive uint64 range")
        normalized.append(value)
    return np.asarray(normalized, dtype=np.uint64)


def _positive_int(values: object, label: str) -> np.ndarray:
    arr = _positive_uint64(values, label)
    if np.any(arr > np.iinfo(np.int64).max):
        raise RuntimeError(f"{label} outside signed int64 range")
    return arr.astype(np.int64)


def _unit_interval(values: object, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise RuntimeError(f"{label} must be a finite one-dimensional real array")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise RuntimeError(f"{label} values must lie in [0,1]")
    return arr


def load_verified_tuning(tuning_raw: bytes) -> TuningData:
    value = _restricted_load(tuning_raw)
    required = ("segment_id", "scan_id", "osi", "osi_p")
    missing = [key for key in required if key not in value]
    if missing:
        raise RuntimeError(f"EASETuning.pkl missing required keys: {missing}")

    segment_id = _positive_uint64(value["segment_id"], "segment_id")
    scan_id = _positive_int(value["scan_id"], "scan_id")
    osi = _unit_interval(value["osi"], "osi")
    osi_p = _unit_interval(value["osi_p"], "osi_p")
    lengths = {len(segment_id), len(scan_id), len(osi), len(osi_p)}
    if len(lengths) != 1:
        raise RuntimeError("EASETuning arrays have unequal lengths")
    if len(segment_id) != len(np.unique(segment_id)):
        raise RuntimeError("duplicate EASETuning segment IDs are a hard stop")
    return TuningData(segment_id=segment_id, scan_id=scan_id, osi=osi, osi_p=osi_p)


def _assign_distance_bins(edges: list[EligibleEdge]) -> list[EligibleEdge]:
    by_fold: dict[int, list[int]] = defaultdict(list)
    for idx, edge in enumerate(edges):
        by_fold[edge.fold].append(idx)

    mutable = [edge.__dict__.copy() for edge in edges]
    for fold, indices in sorted(by_fold.items()):
        ordered = sorted(
            indices,
            key=lambda i: (edges[i].distance_nm, edges[i].source, edges[i].target),
        )
        n_bins = min(10, len(ordered))
        for bin_id, part in enumerate(np.array_split(np.asarray(ordered, dtype=int), n_bins)):
            for index in part.tolist():
                mutable[index]["distance_bin"] = int(bin_id)
    return [EligibleEdge(**row) for row in mutable]


def _mark_quartiles(edges: list[EligibleEdge]) -> list[EligibleEdge]:
    by_fold: dict[int, list[int]] = defaultdict(list)
    for idx, edge in enumerate(edges):
        by_fold[edge.fold].append(idx)
    mutable = [edge.__dict__.copy() for edge in edges]
    for fold, indices in sorted(by_fold.items()):
        ordered = sorted(
            indices,
            key=lambda i: (edges[i].surprise_bits, edges[i].source, edges[i].target),
        )
        k = max(1, len(ordered) // 4)
        for index in ordered[:k]:
            mutable[index]["low"] = True
        for index in ordered[-k:]:
            mutable[index]["high"] = True
    return [EligibleEdge(**row) for row in mutable]


def build_heldout_functional_edges(data: PublicL23Data, tuning: TuningData) -> tuple[list[EligibleEdge], list[float]]:
    if len(data.node_ids) != EXPECTED_STRUCTURAL_NODES:
        raise RuntimeError("unexpected structural node count")
    lookup = {
        int(seg): (int(scan), float(osi), float(osi_p))
        for seg, scan, osi, osi_p in zip(
            tuning.segment_id, tuning.scan_id, tuning.osi, tuning.osi_p, strict=True
        )
    }
    sample = data.sample
    slabs = contiguous_node_slabs(data.node_ids, data.coordinates_nm, n_splits=5, axis=0)
    eligible: list[EligibleEdge] = []
    selected_l2: list[float] = []

    for outer_index in range(5):
        available = [index for index in range(5) if index != outer_index]
        train_nodes = np.concatenate([slabs[index] for index in available])
        test_nodes = slabs[outer_index]
        train = _sample_for_nodes(sample, train_nodes)
        test = _sample_for_nodes(sample, test_nodes)
        l2, _ = _select_l2(sample, slabs, available, "spatial", (1.0, 10.0, 100.0))
        selected_l2.append(float(l2))
        model = GeometryLogisticModel(feature_set="spatial", l2=l2).fit(train)
        probabilities = np.asarray(model.predict_proba(test), dtype=float)
        if not model.converged_:
            raise RuntimeError("spatial model did not converge")
        if not np.all(np.isfinite(probabilities)) or np.any((probabilities <= 0) | (probabilities >= 1)):
            raise RuntimeError("invalid held-out spatial probabilities")

        source = np.asarray(test.source)
        target = np.asarray(test.target)
        distance = np.asarray(test.distance, dtype=float)
        connected = np.asarray(test.connected, dtype=np.int8)
        for i in np.flatnonzero(connected == 1):
            src = int(source[i])
            dst = int(target[i])
            if src not in lookup or dst not in lookup:
                continue
            src_scan, src_osi, src_p = lookup[src]
            dst_scan, dst_osi, dst_p = lookup[dst]
            probability = float(probabilities[i])
            eligible.append(
                EligibleEdge(
                    fold=outer_index,
                    source=src,
                    target=dst,
                    distance_nm=float(distance[i]),
                    probability=probability,
                    surprise_bits=float(-math.log2(probability)),
                    source_scan=src_scan,
                    target_scan=dst_scan,
                    source_osi=src_osi,
                    target_osi=dst_osi,
                    source_osi_p=src_p,
                    target_osi_p=dst_p,
                )
            )

    folds_with_edges = {edge.fold for edge in eligible}
    if len(eligible) < 30 or len(folds_with_edges) < 3:
        raise RuntimeError(
            f"Experiment 013 adequacy stop: {len(eligible)} eligible edges in "
            f"{len(folds_with_edges)} folds"
        )
    return _mark_quartiles(_assign_distance_bins(eligible)), selected_l2


def _mean_difference(edges: Sequence[EligibleEdge]) -> float:
    if not edges:
        raise RuntimeError("cannot score an empty edge set")
    return float(np.mean([edge.orientation_difference for edge in edges]))


def _concordance(edges: Sequence[EligibleEdge]) -> float:
    return float(
        np.mean(
            [
                (edge.source_osi_p < 0.01) == (edge.target_osi_p < 0.01)
                for edge in edges
            ]
        )
    )


def matched_random_test(edges: list[EligibleEdge]) -> dict[str, object]:
    high = [edge for edge in edges if edge.high]
    low = [edge for edge in edges if edge.low]
    if not high or not low:
        raise RuntimeError("high/low residual sets are empty")

    pools: dict[tuple[int, int, bool], list[EligibleEdge]] = defaultdict(list)
    requested: dict[tuple[int, int, bool], int] = defaultdict(int)
    for edge in edges:
        pools[(edge.fold, edge.distance_bin, edge.same_scan)].append(edge)
    for edge in high:
        requested[(edge.fold, edge.distance_bin, edge.same_scan)] += 1
    for key, count in requested.items():
        if len(pools[key]) < count:
            raise RuntimeError(f"insufficient matching stratum {key}")

    rng = np.random.default_rng(RNG_SEED)
    random_means = np.empty(RANDOM_REPLICATES, dtype=float)
    random_concordance = np.empty(RANDOM_REPLICATES, dtype=float)
    strata = sorted(requested)
    for replicate in range(RANDOM_REPLICATES):
        chosen: list[EligibleEdge] = []
        for key in strata:
            pool = pools[key]
            count = requested[key]
            indices = rng.choice(len(pool), size=count, replace=False)
            chosen.extend(pool[int(index)] for index in np.asarray(indices).tolist())
        random_means[replicate] = _mean_difference(chosen)
        random_concordance[replicate] = _concordance(chosen)

    high_mean = _mean_difference(high)
    random_median = float(np.median(random_means))
    if random_median <= 0:
        raise RuntimeError("random median orientation difference is non-positive")
    improvement = (random_median - high_mean) / random_median
    empirical_p = float((1 + int(np.sum(random_means <= high_mean))) / (RANDOM_REPLICATES + 1))
    criterion = bool(improvement >= PRIMARY_MIN_IMPROVEMENT and empirical_p <= PRIMARY_MAX_P)

    tuned_high = [
        edge for edge in high if edge.source_osi_p < 0.01 and edge.target_osi_p < 0.01
    ]
    tuned_only = (
        {"count": len(tuned_high), "mean_osi_difference": _mean_difference(tuned_high)}
        if len(tuned_high) >= 10
        else {"count": len(tuned_high), "mean_osi_difference": None}
    )

    return {
        "high_edge_count": len(high),
        "low_edge_count": len(low),
        "high_mean_osi_difference": high_mean,
        "low_mean_osi_difference": _mean_difference(low),
        "high_minus_low_mean_difference": high_mean - _mean_difference(low),
        "random_median_osi_difference": random_median,
        "random_mean_osi_difference": float(np.mean(random_means)),
        "random_p05_osi_difference": float(np.quantile(random_means, 0.05)),
        "random_p95_osi_difference": float(np.quantile(random_means, 0.95)),
        "relative_improvement_vs_random_median": float(improvement),
        "empirical_one_sided_p": empirical_p,
        "primary_min_improvement": PRIMARY_MIN_IMPROVEMENT,
        "primary_max_p": PRIMARY_MAX_P,
        "primary_criterion_met": criterion,
        "high_tuned_status_concordance": _concordance(high),
        "low_tuned_status_concordance": _concordance(low),
        "random_median_tuned_status_concordance": float(np.median(random_concordance)),
        "both_significantly_tuned_high": tuned_only,
        "matching_strata": {
            f"{fold}:{distance_bin}:{int(same_scan)}": {
                "pool": len(pools[(fold, distance_bin, same_scan)]),
                "requested": requested[(fold, distance_bin, same_scan)],
            }
            for fold, distance_bin, same_scan in strata
        },
    }


def _edge_csv(edges: list[EligibleEdge]) -> bytes:
    stream = io.StringIO(newline="")
    fields = [
        "fold", "source", "target", "distance_nm", "probability", "surprise_bits",
        "source_scan", "target_scan", "source_osi", "target_osi", "source_osi_p",
        "target_osi_p", "distance_bin", "same_scan", "high", "low",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for edge in sorted(edges, key=lambda item: (item.fold, item.source, item.target)):
        writer.writerow(
            {
                "fold": edge.fold,
                "source": edge.source,
                "target": edge.target,
                "distance_nm": f"{edge.distance_nm:.12g}",
                "probability": f"{edge.probability:.17g}",
                "surprise_bits": f"{edge.surprise_bits:.17g}",
                "source_scan": edge.source_scan,
                "target_scan": edge.target_scan,
                "source_osi": f"{edge.source_osi:.17g}",
                "target_osi": f"{edge.target_osi:.17g}",
                "source_osi_p": f"{edge.source_osi_p:.17g}",
                "target_osi_p": f"{edge.target_osi_p:.17g}",
                "distance_bin": edge.distance_bin,
                "same_scan": int(edge.same_scan),
                "high": int(edge.high),
                "low": int(edge.low),
            }
        )
    return stream.getvalue().encode("ascii")


def run_experiment_013(
    *, functional_archive: str | Path, soma_csv: str | Path, synapse_csv: str | Path
) -> tuple[dict[str, object], bytes]:
    archive_raw = Path(functional_archive).read_bytes()
    tuning_raw, functional_source = extract_verified_tuning_pickle(archive_raw)
    tuning = load_verified_tuning(tuning_raw)
    structural = load_public_l23_candidate_data(soma_csv, synapse_csv)
    edges, selected_l2 = build_heldout_functional_edges(structural, tuning)
    primary = matched_random_test(edges)
    edge_csv = _edge_csv(edges)
    high_csv = _edge_csv([edge for edge in edges if edge.high])

    structural_ids = {int(value) for value in structural.node_ids.tolist()}
    functional_ids = {int(value) for value in tuning.segment_id.tolist()}
    per_fold = {
        str(fold): {
            "eligible": sum(edge.fold == fold for edge in edges),
            "high": sum(edge.fold == fold and edge.high for edge in edges),
            "low": sum(edge.fold == fold and edge.low for edge in edges),
        }
        for fold in range(5)
    }
    report: dict[str, object] = {
        "experiment": EXPERIMENT,
        "preregistration_issue": PREREGISTRATION_ISSUE,
        "evidence_level": EVIDENCE_LEVEL,
        "allen_reference_commit": ALLEN_REFERENCE_COMMIT,
        "functional_source": functional_source,
        "structural_source": {
            "soma_md5": SOMA_MD5,
            "synapse_md5": SYNAPSE_MD5,
            "node_count": len(structural.node_ids),
        },
        "functional_row_count": len(tuning.segment_id),
        "functional_ids_in_structural_graph": len(functional_ids & structural_ids),
        "eligible_edge_count": len(edges),
        "eligible_folds": sorted({edge.fold for edge in edges}),
        "selected_spatial_l2_by_fold": selected_l2,
        "per_fold": per_fold,
        "primary": primary,
        "eligible_edge_table_sha256": _sha256(edge_csv),
        "high_residual_manifest_sha256": _sha256(high_csv),
        "ease_tuning_opened": True,
        "ease_trace_opened": False,
        "stimulus_opened": False,
        "structural_training_used_functional_values": False,
        "structural_ranking_used_functional_values": False,
        "interpretation_boundary": (
            "association pilot only; not causal or functional-preservation evidence"
        ),
    }
    return report, edge_csv
