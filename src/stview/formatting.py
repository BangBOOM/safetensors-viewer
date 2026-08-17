from __future__ import annotations

import json
from typing import Any


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if abs(size) < 1024 or unit == units[-1]:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def human_count(value: int) -> str:
    units = ("", "K", "M", "B", "T")
    size = float(value)
    for unit in units:
        if abs(size) < 1000 or unit == units[-1]:
            return f"{int(size):,}" if not unit else f"{size:.2f}{unit}"
        size /= 1000
    raise AssertionError("unreachable")


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)

