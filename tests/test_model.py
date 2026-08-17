from __future__ import annotations

import json
import struct
from pathlib import Path

from stview.model import filter_tensors, load_model, verify_model


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header = {"__metadata__": {"format": "pt"}}
    data = bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        start = len(data)
        data.extend(payload)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(data)],
        }
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    padding = (8 - len(raw_header) % 8) % 8
    raw_header += b" " * padding
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + data)


def create_model(tmp_path: Path) -> Path:
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    write_safetensors(
        first,
        {
            "blocks.0.weight": ("F32", [2, 2], b"\0" * 16),
            "blocks.0.bias": ("F32", [2], b"\0" * 8),
        },
    )
    write_safetensors(second, {"head.weight": ("I16", [3], b"\0" * 6)})
    index = {
        "metadata": {"total_size": 30},
        "weight_map": {
            "blocks.0.weight": first.name,
            "blocks.0.bias": first.name,
            "head.weight": second.name,
        },
    }
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index))
    return index_path


def test_load_sharded_model(tmp_path: Path) -> None:
    index = create_model(tmp_path)
    model = load_model(index)
    assert len(model.shards) == 2
    assert len(model.tensors) == 3
    assert model.total_params == 9
    assert model.total_size == 30
    assert model.summary()["dtypes"] == {"F32": 2, "I16": 1}


def test_directory_auto_discovers_index(tmp_path: Path) -> None:
    create_model(tmp_path)
    model = load_model(tmp_path)
    assert model.index_path is not None
    assert model.index_path.name == "model.safetensors.index.json"


def test_filters(tmp_path: Path) -> None:
    model = load_model(create_model(tmp_path))
    assert [t.name for t in filter_tensors(model.tensors, match="blocks.*", ndim=1)] == ["blocks.0.bias"]
    assert [t.name for t in filter_tensors(model.tensors, regex=r"head\.", dtype="i16")] == ["head.weight"]


def test_verify_valid_model(tmp_path: Path) -> None:
    model = load_model(create_model(tmp_path))
    assert verify_model(model) == []


def test_verify_detects_index_size_mismatch(tmp_path: Path) -> None:
    index = create_model(tmp_path)
    payload = json.loads(index.read_text())
    payload["metadata"]["total_size"] = 999
    index.write_text(json.dumps(payload))
    issues = verify_model(load_model(index))
    assert any("total_size" in issue.message for issue in issues)

