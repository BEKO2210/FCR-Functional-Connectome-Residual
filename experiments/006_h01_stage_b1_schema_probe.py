"""Experiment 006 Stage B1: bounded H01 synapse-export schema probe.

The probe intentionally reports structure only: no neuron IDs, synapse IDs,
counts, degrees, densities, or model outcomes are emitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_PATHS = (
    ("pre_synaptic_site", "neuron_id"),
    ("post_synaptic_partner", "neuron_id"),
)


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
    text = Path(input_path).read_text(encoding="utf-8", errors="strict")
    record, source_format = decode_first_record(text)
    missing = [".".join(path) for path in REQUIRED_PATHS if not _has_path(record, path)]
    if missing:
        raise RuntimeError(f"H01 schema missing required identity paths: {missing}")

    return {
        "experiment": "006-stage-b1-schema-probe",
        "bounded_probe": True,
        "source_format": source_format,
        "required_identity_paths": [".".join(path) for path in REQUIRED_PATHS],
        "required_identity_paths_present": True,
        "schema": schema_tree(record),
        "values_emitted": False,
        "connectivity_metrics_computed": False,
        "model_metrics_computed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = probe_schema(args.input)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
