"""H01 connectivity adapters for Experiment 006.

All primary node identities are frozen before this module is allowed to inspect
H01 connectivity. Stage B1 verifies the release schema, streams every frozen
shard, and emits deterministic extraction artifacts before any model runs.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import shutil
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_IDENTITY_PATHS = (
    ("pre_synaptic_site", "neuron_id"),
    ("post_synaptic_partner", "neuron_id"),
)
FROZEN_NODE_SHA256 = "3c954b9206f8e14859fb789b11cb008ee2245c3192cfc29793ab13bb231f55f4"
FROZEN_NODE_COUNT = 1_500
EXPECTED_CANDIDATE_PAIRS = FROZEN_NODE_COUNT * (FROZEN_NODE_COUNT - 1)
H01_JSON_BASE_URL = (
    "https://storage.googleapis.com/h01-release/"
    "data/20210601/c3/synapses/exported/json"
)
H01_SHARD_BYTES = (
    757111597, 757457938, 758157092, 757988051, 757519932, 757587361,
    757869883, 757967887, 758094540, 758185812, 758359599, 758298719,
    758198184, 758426537, 758333435, 758311289, 758483572, 758397443,
    758340619, 758311613, 758533141, 758542365, 758548443, 758617170,
    758571039, 758754344, 758752084, 758705210, 758534750, 758751355,
    758677390, 758772750, 758709004, 758870042, 758916058, 759018273,
    758749040, 758739909, 758795521, 759133223, 758958904, 758832599,
    758810767, 759202353, 758912232, 759009555, 758976286, 758938293,
    758929757, 759132767, 758998385, 758918131, 758946738, 758824868,
    758989657, 758944114, 758967357, 759114059, 759174860, 759132512,
    759138746, 759046311, 759135045, 759046846, 759092750, 759207038,
    759204235, 759197188, 759394866, 759366987, 759322580, 759353486,
    759434562, 759260510, 759335486, 759356028, 759331867, 759453190,
    759395162, 759187062, 759412687, 759391333, 759459780, 759558964,
    759412966, 759404188, 759462987, 759431678, 759560489, 759378980,
    759375682, 759410302, 759456366, 759420229, 759437439, 759683315,
    759604619, 759768432, 759656580, 759788990, 759549348, 759375586,
    759794709, 759711607, 759781515, 759757694, 759701105, 759740066,
    759821028, 759811339, 759821725, 759831012, 759790955, 759851325,
    759821365, 759803401, 759740561, 759791436, 759949267, 759943897,
    759782673, 759840759, 759901687, 759953730, 759918914, 760018879,
    759967706, 760003621, 760096560, 760019063, 760061800, 760016424,
    760137689, 760105179, 760147846, 760094163, 760105859, 760194268,
    760074953, 760252320, 760230302, 760380801, 760230032, 760167969,
    760477596, 760700800, 760453930, 760375980, 760423472, 760553020,
    760482453, 760859224, 760686882, 760799657, 760826881, 760664635,
    760832649, 760885135, 760933477, 760947121, 761105546, 761050933,
    761166520, 761265848, 761159613, 761367560,
)
H01_SHARD_COUNT = len(H01_SHARD_BYTES)
H01_TOTAL_BYTES = sum(H01_SHARD_BYTES)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def schema_tree(value: Any) -> Any:
    """Return keys and value types without retaining any source values."""
    if isinstance(value, dict):
        return {
            key: {"type": _type_name(child), "schema": schema_tree(child)}
            for key, child in sorted(value.items())
        }
    if isinstance(value, list):
        if not value:
            return {"item_type": "unknown", "item_schema": None}
        return {
            "item_type": _type_name(value[0]),
            "item_schema": schema_tree(value[0]),
        }
    return None


def _has_path(record: dict[str, Any], path: tuple[str, ...]) -> bool:
    current: Any = record
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return True


def decode_first_record(text: str) -> tuple[dict[str, Any], str]:
    """Decode only the first record from a JSON array, JSONL stream, or object."""
    stripped = text.lstrip("\ufeff \t\r\n")
    if not stripped:
        raise ValueError("schema probe is empty")

    decoder = json.JSONDecoder()
    if stripped.startswith("["):
        payload = stripped[1:].lstrip()
        record, _ = decoder.raw_decode(payload)
        source_format = "json-array"
    elif stripped.startswith("{"):
        record, end = decoder.raw_decode(stripped)
        remainder = stripped[end:].lstrip()
        source_format = "jsonl" if remainder.startswith("{") else "json-object"
    else:
        raise ValueError("unsupported H01 JSON prefix")

    if not isinstance(record, dict):
        raise ValueError("first H01 JSON record is not an object")
    return record, source_format


def probe_schema(input_path: str | Path) -> dict[str, Any]:
    """Inspect one bounded shard prefix and emit schema only, never record values."""
    text = Path(input_path).read_text(encoding="utf-8", errors="strict")
    record, source_format = decode_first_record(text)
    missing = [
        ".".join(path) for path in REQUIRED_IDENTITY_PATHS if not _has_path(record, path)
    ]
    if missing:
        raise RuntimeError(f"H01 schema missing required identity paths: {missing}")

    return {
        "bounded_probe": True,
        "source_format": source_format,
        "required_identity_paths": [".".join(path) for path in REQUIRED_IDENTITY_PATHS],
        "required_identity_paths_present": True,
        "schema": schema_tree(record),
        "values_emitted": False,
        "connectivity_metrics_computed": False,
        "model_metrics_computed": False,
    }


def shard_name(index: int) -> str:
    if not 0 <= index < H01_SHARD_COUNT:
        raise ValueError(f"H01 shard index out of range: {index}")
    return f"export{index:012d}.json"


def shard_url(index: int) -> str:
    return f"{H01_JSON_BASE_URL}/{shard_name(index)}"


def assigned_shards(part_index: int, part_count: int) -> list[int]:
    if part_count <= 0:
        raise ValueError("part_count must be positive")
    if not 0 <= part_index < part_count:
        raise ValueError("part_index must be in [0, part_count)")
    return [index for index in range(H01_SHARD_COUNT) if index % part_count == part_index]


def load_frozen_node_ids(
    node_csv: str | Path,
    *,
    expected_sha256: str = FROZEN_NODE_SHA256,
    expected_count: int = FROZEN_NODE_COUNT,
) -> list[int]:
    source = Path(node_csv)
    observed_hash = sha256_file(source)
    if observed_hash != expected_sha256:
        raise RuntimeError(
            f"frozen H01 node hash mismatch: observed={observed_hash} expected={expected_sha256}"
        )

    frame = pd.read_csv(source)
    if "c3_rep_manual" not in frame.columns:
        raise RuntimeError("frozen H01 node CSV missing c3_rep_manual")
    values = pd.to_numeric(frame["c3_rep_manual"], errors="raise")
    if values.isna().any():
        raise RuntimeError("frozen H01 node CSV contains null identities")
    raw_ids = values.tolist()
    if any(int(value) != value for value in raw_ids):
        raise RuntimeError("frozen H01 node CSV contains non-integer identities")
    ids = [int(value) for value in raw_ids]
    if len(ids) != expected_count:
        raise RuntimeError(f"frozen H01 node count mismatch: {len(ids)} != {expected_count}")
    if len(set(ids)) != len(ids):
        raise RuntimeError("frozen H01 node CSV contains duplicate identities")
    return ids


def coerce_neuron_id(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdecimal():
            return int(stripped)
        return None
    return None


def iter_jsonl_records(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid H01 JSONL record at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise RuntimeError(f"H01 JSONL line {line_number} is not an object")
            yield record


def _record_neuron_id(record: dict[str, Any], side: str) -> int | None:
    site = record.get(side)
    if not isinstance(site, dict):
        return None
    return coerce_neuron_id(site.get("neuron_id"))


def extract_selected_pairs(
    shard_path: str | Path,
    selected_ids: set[int],
) -> tuple[Counter[tuple[int, int]], dict[str, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    records_seen = 0
    missing_or_invalid_identity_records = 0
    selected_synapses = 0
    selected_autapse_synapses = 0

    for record in iter_jsonl_records(shard_path):
        records_seen += 1
        pre = _record_neuron_id(record, "pre_synaptic_site")
        post = _record_neuron_id(record, "post_synaptic_partner")
        if pre is None or post is None:
            missing_or_invalid_identity_records += 1
            continue
        if pre not in selected_ids or post not in selected_ids:
            continue
        counts[(pre, post)] += 1
        selected_synapses += 1
        if pre == post:
            selected_autapse_synapses += 1

    diagnostics = {
        "records_seen": records_seen,
        "missing_or_invalid_identity_records": missing_or_invalid_identity_records,
        "selected_synapses": selected_synapses,
        "selected_autapse_synapses": selected_autapse_synapses,
        "selected_ordered_pairs_including_autapses": len(counts),
    }
    return counts, diagnostics


def download_shard(
    index: int,
    destination: str | Path,
    *,
    retries: int = 5,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = H01_SHARD_BYTES[index]
    url = shard_url(index)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        temp = target.with_suffix(target.suffix + ".part")
        temp.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "FCR-H01-Stage-B1/1.0"})
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                with temp.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            observed_bytes = temp.stat().st_size
            if observed_bytes != expected_bytes:
                raise RuntimeError(
                    f"H01 shard {index} size mismatch: {observed_bytes} != {expected_bytes}"
                )
            temp.replace(target)
            return {
                "shard_index": index,
                "name": shard_name(index),
                "expected_bytes": expected_bytes,
                "observed_bytes": observed_bytes,
                "download_attempt": attempt,
            }
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = exc
            temp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 16))

    message = f"failed to download H01 shard {index} after {retries} attempts"
    raise RuntimeError(message) from last_error


def write_pair_counts(
    counts: Counter[tuple[int, int]],
    destination: str | Path,
) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["pre_id", "post_id", "synapse_count"])
        for (pre, post), count in sorted(counts.items()):
            writer.writerow([pre, post, count])


def read_pair_counts(paths: Iterable[str | Path]) -> Counter[tuple[int, int]]:
    combined: Counter[tuple[int, int]] = Counter()
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["pre_id", "post_id", "synapse_count"]:
                raise RuntimeError(f"unexpected partial edge schema in {path}")
            for row in reader:
                pre = int(row["pre_id"])
                post = int(row["post_id"])
                count = int(row["synapse_count"])
                if count <= 0:
                    raise RuntimeError(f"non-positive synapse count in {path}")
                combined[(pre, post)] += count
    return combined


def _open_deterministic_gzip_text(path: str | Path) -> io.TextIOWrapper:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = target.open("wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return io.TextIOWrapper(gz, encoding="utf-8", newline="")


def write_sparse_edges_gzip(
    counts: Counter[tuple[int, int]],
    destination: str | Path,
) -> dict[str, int]:
    nonself = [((pre, post), count) for (pre, post), count in counts.items() if pre != post]
    autapse_synapses = sum(count for (pre, post), count in counts.items() if pre == post)
    with _open_deterministic_gzip_text(destination) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["pre_id", "post_id", "synapse_count"])
        for (pre, post), count in sorted(nonself):
            writer.writerow([pre, post, count])
    return {
        "connected_nonself_pairs": len(nonself),
        "selected_nonself_synapses": sum(count for _, count in nonself),
        "selected_autapse_synapses": autapse_synapses,
    }


def write_candidate_table_gzip(
    node_ids: list[int],
    counts: Counter[tuple[int, int]],
    destination: str | Path,
) -> dict[str, int]:
    if len(node_ids) != len(set(node_ids)):
        raise RuntimeError("candidate table node IDs are not unique")
    expected_rows = len(node_ids) * (len(node_ids) - 1)
    rows = 0
    positives = 0

    with _open_deterministic_gzip_text(destination) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["pre_id", "post_id", "synapse_count", "connected"])
        for pre in node_ids:
            for post in node_ids:
                if pre == post:
                    continue
                multiplicity = counts.get((pre, post), 0)
                connected = int(multiplicity > 0)
                positives += connected
                rows += 1
                writer.writerow([pre, post, multiplicity, connected])

    if rows != expected_rows:
        raise RuntimeError(f"candidate row count mismatch: {rows} != {expected_rows}")
    return {"candidate_rows": rows, "connected_nonself_pairs": positives}
