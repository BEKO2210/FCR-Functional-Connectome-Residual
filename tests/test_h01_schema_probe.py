from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import _006_h01_stage_b1_schema_probe as probe_module


def _record() -> dict[str, object]:
    return {
        "location": {"x": 1, "y": 2, "z": 3},
        "pre_synaptic_site": {
            "id": 111,
            "neuron_id": 222,
            "centroid": {"x": 4, "y": 5, "z": 6},
        },
        "post_synaptic_partner": {
            "id": 333,
            "neuron_id": 444,
            "centroid": {"x": 7, "y": 8, "z": 9},
        },
    }


def test_decode_first_record_accepts_jsonl() -> None:
    text = json.dumps(_record()) + "\n" + json.dumps(_record()) + "\n"
    record, source_format = probe_module.decode_first_record(text)
    assert source_format == "jsonl"
    assert record["pre_synaptic_site"]["neuron_id"] == 222


def test_decode_first_record_accepts_array() -> None:
    text = json.dumps([_record(), _record()])
    record, source_format = probe_module.decode_first_record(text)
    assert source_format == "json-array"
    assert record["post_synaptic_partner"]["neuron_id"] == 444


def test_probe_schema_emits_structure_not_values(tmp_path: Path) -> None:
    source = tmp_path / "probe.json"
    source.write_text(json.dumps(_record()) + "\n" + json.dumps(_record()), encoding="utf-8")
    report = probe_module.probe_schema(source)
    rendered = json.dumps(report)

    assert report["required_identity_paths_present"] is True
    assert report["values_emitted"] is False
    assert '"222"' not in rendered
    assert '"444"' not in rendered
    assert report["schema"]["pre_synaptic_site"]["schema"]["neuron_id"]["type"] == "int"


def test_probe_schema_rejects_missing_identity_path(tmp_path: Path) -> None:
    record = _record()
    del record["post_synaptic_partner"]["neuron_id"]
    source = tmp_path / "probe.json"
    source.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RuntimeError, match="post_synaptic_partner.neuron_id"):
        probe_module.probe_schema(source)
