from __future__ import annotations

import fnmatch
import json
import math
import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


MAX_HEADER_SIZE = 100 * 1024 * 1024

# Safetensors stores dtype names in the file header. Values are bits per element.
DTYPE_BITS: dict[str, int] = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "I64": 64,
    "U64": 64,
    "F64": 64,
    "I4": 4,
    "U4": 4,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
}


class InspectError(Exception):
    """Raised when an input cannot be resolved or parsed."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    offsets: tuple[int, int]
    shard: Path

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def size_bytes(self) -> int:
        return self.offsets[1] - self.offsets[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "rank": self.rank,
            "dtype": self.dtype,
            "params": self.numel,
            "size_bytes": self.size_bytes,
            "shard": self.shard.name,
            "path": str(self.shard),
            "data_offsets": list(self.offsets),
        }


@dataclass
class ShardInfo:
    path: Path
    header_size: int
    file_size: int
    metadata: dict[str, str] | None
    tensors: list[TensorInfo]

    @property
    def data_size(self) -> int:
        return sum(t.size_bytes for t in self.tensors)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.path.name,
            "path": str(self.path),
            "file_size": self.file_size,
            "header_size": self.header_size,
            "data_size": self.data_size,
            "tensor_count": len(self.tensors),
            "metadata": self.metadata,
        }


@dataclass
class ModelInfo:
    source: Path
    shards: list[ShardInfo]
    index_path: Path | None = None
    index_metadata: dict[str, Any] = field(default_factory=dict)
    weight_map: dict[str, str] | None = None

    @property
    def tensors(self) -> list[TensorInfo]:
        return [tensor for shard in self.shards for tensor in shard.tensors]

    @property
    def total_params(self) -> int:
        return sum(t.numel for t in self.tensors)

    @property
    def total_size(self) -> int:
        return sum(t.size_bytes for t in self.tensors)

    @property
    def disk_size(self) -> int:
        return sum(s.file_size for s in self.shards)

    def tensor_map(self) -> dict[str, TensorInfo]:
        return {tensor.name: tensor for tensor in self.tensors}

    def summary(self) -> dict[str, Any]:
        dtypes = Counter(t.dtype for t in self.tensors)
        ranks = Counter(str(t.rank) for t in self.tensors)
        return {
            "source": str(self.source),
            "index": str(self.index_path) if self.index_path else None,
            "tensor_count": len(self.tensors),
            "total_params": self.total_params,
            "total_size_bytes": self.total_size,
            "disk_size_bytes": self.disk_size,
            "shard_count": len(self.shards),
            "dtypes": dict(sorted(dtypes.items())),
            "ranks": dict(sorted(ranks.items(), key=lambda item: int(item[0]))),
            "index_metadata": self.index_metadata,
            "shards": [shard.as_dict() for shard in self.shards],
        }


@dataclass(frozen=True)
class VerificationIssue:
    level: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"level": self.level, "message": self.message}


def read_header(path: Path) -> ShardInfo:
    """Read only the JSON header of a safetensors file."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise InspectError(f"{path}: file is too short for a safetensors header")
            header_size = struct.unpack("<Q", raw_length)[0]
            if header_size > MAX_HEADER_SIZE:
                raise InspectError(
                    f"{path}: header is too large ({header_size} bytes, max {MAX_HEADER_SIZE})"
                )
            if header_size > file_size - 8:
                raise InspectError(f"{path}: header extends beyond the end of the file")
            raw_header = stream.read(header_size)
    except OSError as exc:
        raise InspectError(f"cannot read {path}: {exc}") from exc

    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InspectError(f"{path}: invalid safetensors JSON header: {exc}") from exc
    if not isinstance(header, dict):
        raise InspectError(f"{path}: safetensors header must be a JSON object")

    metadata = header.pop("__metadata__", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise InspectError(f"{path}: __metadata__ must be an object")

    tensors: list[TensorInfo] = []
    for name, value in header.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise InspectError(f"{path}: invalid tensor entry {name!r}")
        try:
            dtype = value["dtype"]
            shape = tuple(value["shape"])
            offsets = tuple(value["data_offsets"])
        except (KeyError, TypeError) as exc:
            raise InspectError(f"{path}: malformed metadata for tensor {name!r}") from exc
        if not isinstance(dtype, str):
            raise InspectError(f"{path}: tensor {name!r} has an invalid dtype")
        if any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape):
            raise InspectError(f"{path}: tensor {name!r} has an invalid shape")
        if (
            len(offsets) != 2
            or any(not isinstance(offset, int) or isinstance(offset, bool) for offset in offsets)
        ):
            raise InspectError(f"{path}: tensor {name!r} has invalid data offsets")
        tensors.append(TensorInfo(name, shape, dtype, offsets, path))

    return ShardInfo(path, header_size, file_size, metadata, tensors)


def _read_index(index_path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InspectError(f"cannot read index {index_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("weight_map"), dict):
        raise InspectError(f"{index_path}: expected a Hugging Face weight_map object")
    weight_map = payload["weight_map"]
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in weight_map.items()):
        raise InspectError(f"{index_path}: weight_map keys and values must be strings")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise InspectError(f"{index_path}: metadata must be an object")
    return weight_map, metadata


def resolve_input(source: Path) -> tuple[list[Path], Path | None, dict[str, str] | None, dict[str, Any]]:
    source = source.expanduser().resolve()
    if not source.exists():
        raise InspectError(f"input does not exist: {source}")

    index_path: Path | None = None
    weight_map: dict[str, str] | None = None
    index_metadata: dict[str, Any] = {}

    if source.is_dir():
        indexes = sorted(source.glob("*.safetensors.index.json"))
        if len(indexes) > 1:
            names = ", ".join(path.name for path in indexes)
            raise InspectError(f"multiple index files found; pass one explicitly: {names}")
        if indexes:
            index_path = indexes[0]
        else:
            shards = sorted(source.glob("*.safetensors"))
            if not shards:
                raise InspectError(f"no .safetensors files found in {source}")
            return shards, None, None, {}
    elif source.name.endswith(".safetensors.index.json") or source.suffix == ".json":
        index_path = source
    elif source.suffix == ".safetensors":
        return [source], None, None, {}
    else:
        raise InspectError("input must be a directory, .safetensors file, or index JSON")

    assert index_path is not None
    weight_map, index_metadata = _read_index(index_path)
    shard_names = sorted(set(weight_map.values()), key=natural_key)
    shards = [index_path.parent / name for name in shard_names]
    return shards, index_path, weight_map, index_metadata


def load_model(source: Path) -> ModelInfo:
    source = source.expanduser().resolve()
    shards, index_path, weight_map, index_metadata = resolve_input(source)
    missing = [path for path in shards if not path.is_file()]
    if missing:
        raise InspectError("missing shard(s): " + ", ".join(str(path) for path in missing))
    shard_infos = [read_header(path) for path in shards]
    return ModelInfo(source, shard_infos, index_path, index_metadata, weight_map)


def natural_key(value: str | Path) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", str(value)))


def filter_tensors(
    tensors: Iterable[TensorInfo],
    *,
    match: str | None = None,
    regex: str | None = None,
    dtype: str | None = None,
    ndim: int | None = None,
    shard: str | None = None,
) -> list[TensorInfo]:
    pattern = re.compile(regex) if regex else None
    result = []
    for tensor in tensors:
        if match and not fnmatch.fnmatchcase(tensor.name, match):
            continue
        if pattern and not pattern.search(tensor.name):
            continue
        if dtype and tensor.dtype.casefold() != dtype.casefold():
            continue
        if ndim is not None and tensor.rank != ndim:
            continue
        if shard and not fnmatch.fnmatchcase(tensor.shard.name, shard):
            continue
        result.append(tensor)
    return result


def verify_model(model: ModelInfo) -> list[VerificationIssue]:
    issues: list[VerificationIssue] = []
    seen: dict[str, Path] = {}

    for shard in model.shards:
        data_capacity = shard.file_size - 8 - shard.header_size
        ordered = sorted(shard.tensors, key=lambda tensor: tensor.offsets)
        previous_end = 0
        for tensor in ordered:
            start, end = tensor.offsets
            if tensor.name in seen:
                issues.append(
                    VerificationIssue(
                        "error", f"duplicate tensor {tensor.name!r} in {seen[tensor.name].name} and {shard.path.name}"
                    )
                )
            else:
                seen[tensor.name] = shard.path
            if start < 0 or end < start:
                issues.append(VerificationIssue("error", f"{tensor.name}: invalid offsets {tensor.offsets}"))
                continue
            if start < previous_end:
                issues.append(VerificationIssue("error", f"{tensor.name}: data overlaps a previous tensor"))
            if end > data_capacity:
                issues.append(VerificationIssue("error", f"{tensor.name}: data extends beyond {shard.path.name}"))
            previous_end = max(previous_end, end)

            bits = DTYPE_BITS.get(tensor.dtype)
            if bits is None:
                issues.append(VerificationIssue("warning", f"{tensor.name}: unknown dtype {tensor.dtype!r}"))
            else:
                expected = (tensor.numel * bits + 7) // 8
                if expected != tensor.size_bytes:
                    issues.append(
                        VerificationIssue(
                            "error",
                            f"{tensor.name}: shape/dtype requires {expected} bytes, offsets contain {tensor.size_bytes}",
                        )
                    )

    if model.weight_map is not None:
        actual = model.tensor_map()
        for name, expected_shard in model.weight_map.items():
            tensor = actual.get(name)
            if tensor is None:
                issues.append(VerificationIssue("error", f"index tensor {name!r} is missing"))
            elif tensor.shard.name != expected_shard:
                issues.append(
                    VerificationIssue(
                        "error",
                        f"index maps {name!r} to {expected_shard}, found in {tensor.shard.name}",
                    )
                )
        for name in sorted(set(actual) - set(model.weight_map), key=natural_key):
            issues.append(VerificationIssue("error", f"tensor {name!r} is not present in the index"))

        declared_size = model.index_metadata.get("total_size")
        if isinstance(declared_size, int) and declared_size != model.total_size:
            issues.append(
                VerificationIssue(
                    "error",
                    f"index total_size is {declared_size}, headers contain {model.total_size} bytes",
                )
            )

    # Ask the official library to validate the format too, without loading tensors.
    try:
        from safetensors import safe_open

        for shard in model.shards:
            with safe_open(shard.path, framework="numpy", device="cpu") as handle:
                list(handle.keys())
    except Exception as exc:  # safetensors exposes several backend-specific error types
        issues.append(VerificationIssue("error", f"safetensors validation failed: {exc}"))

    return issues
