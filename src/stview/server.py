from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Path as ApiPath, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, Response

from . import __version__
from .model import ModelInfo, TensorInfo, load_model, natural_key, verify_model


ASSETS = files("stview.web_assets")


def create_app(source: Path | None = None, *, model: ModelInfo | None = None) -> FastAPI:
    """Create a web app pinned to one local model."""
    if model is None:
        if source is None:
            raise ValueError("source or model is required")
        model = load_model(source)

    app = FastAPI(
        title="Safetensors Viewer",
        description="Browse safetensors header metadata without loading tensor values.",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.model = model
    app.state.tensor_map = model.tensor_map()

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return ASSETS.joinpath("index.html").read_text(encoding="utf-8")

    @app.get("/assets/app.css", include_in_schema=False)
    async def css() -> Response:
        return Response(
            ASSETS.joinpath("app.css").read_text(encoding="utf-8"),
            media_type="text/css",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/assets/app.js", include_in_schema=False)
    async def javascript() -> Response:
        return Response(
            ASSETS.joinpath("app.js").read_text(encoding="utf-8"),
            media_type="text/javascript",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/info")
    async def info() -> dict[str, Any]:
        return model.summary()

    @app.get("/api/tensors")
    async def tensors(
        q: Annotated[str | None, Query(description="Case-insensitive name substring")] = None,
        dtype: Annotated[str | None, Query(description="Exact dtype")] = None,
        rank: Annotated[int | None, Query(ge=0, description="Exact tensor rank")] = None,
        shard: Annotated[str | None, Query(description="Exact shard filename")] = None,
        sort: Annotated[Literal["name", "size", "params", "dtype", "rank"], Query()] = "name",
        order: Annotated[Literal["asc", "desc"], Query()] = "asc",
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=500)] = 50,
    ) -> dict[str, Any]:
        selected = model.tensors
        if q:
            needle = q.casefold()
            selected = [tensor for tensor in selected if needle in tensor.name.casefold()]
        if dtype:
            selected = [tensor for tensor in selected if tensor.dtype.casefold() == dtype.casefold()]
        if rank is not None:
            selected = [tensor for tensor in selected if tensor.rank == rank]
        if shard:
            selected = [tensor for tensor in selected if tensor.shard.name == shard]

        sorters = {
            "name": lambda tensor: natural_key(tensor.name),
            "size": lambda tensor: (tensor.size_bytes, natural_key(tensor.name)),
            "params": lambda tensor: (tensor.numel, natural_key(tensor.name)),
            "dtype": lambda tensor: (tensor.dtype, natural_key(tensor.name)),
            "rank": lambda tensor: (tensor.rank, natural_key(tensor.name)),
        }
        selected.sort(key=sorters[sort], reverse=order == "desc")
        page = selected[offset : offset + limit]
        return {
            "items": [tensor.as_dict() for tensor in page],
            "total": len(selected),
            "offset": offset,
            "limit": limit,
        }

    @app.get("/api/tensors/{tensor_name:path}")
    async def tensor_detail(
        tensor_name: Annotated[str, ApiPath(description="Exact tensor name")],
    ) -> dict[str, Any]:
        tensor = app.state.tensor_map.get(tensor_name)
        if tensor is None:
            raise HTTPException(status_code=404, detail="Tensor not found")
        return tensor.as_dict()

    @app.get("/api/modules")
    async def modules(
        depth: Annotated[int, Query(ge=1, le=8)] = 2,
    ) -> list[dict[str, Any]]:
        return module_summary(model.tensors, depth)

    @app.post("/api/verify")
    async def verify() -> dict[str, Any]:
        issues = await run_in_threadpool(verify_model, model)
        errors = sum(issue.level == "error" for issue in issues)
        warnings = sum(issue.level == "warning" for issue in issues)
        return {
            "ok": errors == 0,
            "errors": errors,
            "warnings": warnings,
            "tensor_count": len(model.tensors),
            "shard_count": len(model.shards),
            "issues": [issue.as_dict() for issue in issues],
        }

    return app


def module_summary(tensors: list[TensorInfo], depth: int) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, int]] = {}
    for tensor in tensors:
        parts = tensor.name.split(".")
        name = ".".join(parts[:depth])
        group = groups.setdefault(name, {"tensor_count": 0, "params": 0, "size_bytes": 0})
        group["tensor_count"] += 1
        group["params"] += tensor.numel
        group["size_bytes"] += tensor.size_bytes
    return [
        {"name": name, **values}
        for name, values in sorted(groups.items(), key=lambda item: natural_key(item[0]))
    ]

