from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from . import __version__
from .formatting import human_bytes, human_count, json_text
from .model import (
    InspectError,
    ModelInfo,
    TensorInfo,
    filter_tensors,
    load_model,
    natural_key,
    verify_model,
)


app = typer.Typer(
    name="stview",
    help="Inspect Hugging Face safetensors weights without loading tensor data.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()
err_console = Console(stderr=True)
SourceArg = Annotated[Path, typer.Argument(help="Model directory, index JSON, or .safetensors file")]


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"stview {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=version_callback, is_eager=True, help="Show version and exit"),
    ] = None,
) -> None:
    """Inspect Hugging Face safetensors weights."""


def get_model(source: Path) -> ModelInfo:
    try:
        return load_model(source)
    except InspectError as exc:
        err_console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(2) from exc


@app.command("info")
def info_command(
    source: SourceArg,
    output: Annotated[str, typer.Option("--format", "-f", help="Output format: table or json")] = "table",
) -> None:
    """Show a model-level summary and shard breakdown."""
    model = get_model(source)
    summary = model.summary()
    if output == "json":
        typer.echo(json_text(summary))
        return
    if output != "table":
        raise_output_error(output, ("table", "json"))

    overview = Table(title="Safetensors model", show_header=False, box=None)
    overview.add_column(style="bold cyan")
    overview.add_column()
    overview.add_row("Source", str(model.source))
    overview.add_row("Index", str(model.index_path) if model.index_path else "—")
    overview.add_row("Tensors", f"{len(model.tensors):,}")
    overview.add_row("Parameters", f"{model.total_params:,} ({human_count(model.total_params)})")
    overview.add_row("Tensor data", f"{model.total_size:,} bytes ({human_bytes(model.total_size)})")
    overview.add_row("Disk size", human_bytes(model.disk_size))
    overview.add_row("Shards", str(len(model.shards)))
    overview.add_row("Dtypes", ", ".join(f"{key}: {value:,}" for key, value in summary["dtypes"].items()))
    overview.add_row("Ranks", ", ".join(f"{key}D: {value:,}" for key, value in summary["ranks"].items()))
    console.print(overview)

    shards = Table(title="Shards")
    shards.add_column("Shard", style="cyan")
    shards.add_column("Tensors", justify="right")
    shards.add_column("Data", justify="right")
    shards.add_column("File", justify="right")
    shards.add_column("Metadata")
    for shard in model.shards:
        shards.add_row(
            shard.path.name,
            f"{len(shard.tensors):,}",
            human_bytes(shard.data_size),
            human_bytes(shard.file_size),
            json.dumps(shard.metadata, ensure_ascii=False) if shard.metadata else "—",
        )
    console.print(shards)


@app.command("list")
def list_command(
    source: SourceArg,
    match: Annotated[str | None, typer.Option(help="Filter names with a shell-style glob")] = None,
    regex: Annotated[str | None, typer.Option(help="Filter names with a regular expression")] = None,
    dtype: Annotated[str | None, typer.Option(help="Filter by dtype, e.g. F32 or BF16")] = None,
    ndim: Annotated[int | None, typer.Option(help="Filter by tensor rank")] = None,
    shard: Annotated[str | None, typer.Option(help="Filter shard names with a shell-style glob")] = None,
    sort: Annotated[str, typer.Option(help="Sort by name, size, params, dtype, or rank")] = "name",
    desc: Annotated[bool, typer.Option(help="Reverse the sort order")] = False,
    limit: Annotated[int | None, typer.Option(help="Maximum number of rows to emit")] = None,
    output: Annotated[str, typer.Option("--format", "-f", help="Output format: table, json, or csv")] = "table",
) -> None:
    """List tensors with filtering and sorting."""
    model = get_model(source)
    try:
        tensors = filter_tensors(model.tensors, match=match, regex=regex, dtype=dtype, ndim=ndim, shard=shard)
    except re.error as exc:
        err_console.print(f"[bold red]Error:[/] invalid regular expression: {exc}")
        raise typer.Exit(2) from exc

    sorters = {
        "name": lambda tensor: natural_key(tensor.name),
        "size": lambda tensor: tensor.size_bytes,
        "params": lambda tensor: tensor.numel,
        "dtype": lambda tensor: (tensor.dtype, natural_key(tensor.name)),
        "rank": lambda tensor: (tensor.rank, natural_key(tensor.name)),
    }
    if sort not in sorters:
        err_console.print(f"[bold red]Error:[/] unknown sort key {sort!r}")
        raise typer.Exit(2)
    tensors.sort(key=sorters[sort], reverse=desc)
    total_matches = len(tensors)
    if limit is not None:
        if limit < 0:
            err_console.print("[bold red]Error:[/] --limit must be non-negative")
            raise typer.Exit(2)
        tensors = tensors[:limit]

    rows = [tensor.as_dict() for tensor in tensors]
    if output == "json":
        typer.echo(json_text(rows))
    elif output == "csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=("name", "shape", "rank", "dtype", "params", "size_bytes", "shard"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key]) if key == "shape" else row[key] for key in writer.fieldnames})
        typer.echo(buffer.getvalue(), nl=False)
    elif output == "table":
        table = Table(title=f"Tensors ({len(tensors):,} shown / {total_matches:,} matched)")
        table.add_column("Name", style="cyan", overflow="fold")
        table.add_column("Shape")
        table.add_column("Dtype")
        table.add_column("Params", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Shard")
        for tensor in tensors:
            table.add_row(
                tensor.name,
                format_shape(tensor.shape),
                tensor.dtype,
                f"{tensor.numel:,}",
                human_bytes(tensor.size_bytes),
                tensor.shard.name,
            )
        console.print(table)
    else:
        raise_output_error(output, ("table", "json", "csv"))


@app.command("show")
def show_command(
    source: SourceArg,
    name: Annotated[str, typer.Argument(help="Exact tensor name")],
    output: Annotated[str, typer.Option("--format", "-f", help="Output format: table or json")] = "table",
) -> None:
    """Show metadata for one tensor."""
    model = get_model(source)
    tensor = model.tensor_map().get(name)
    if tensor is None:
        suggestions = sorted(
            (item for item in model.tensors if name.casefold() in item.name.casefold()),
            key=lambda item: natural_key(item.name),
        )[:5]
        err_console.print(f"[bold red]Error:[/] tensor not found: {name}")
        if suggestions:
            err_console.print("Possible matches:")
            for item in suggestions:
                err_console.print(f"  {item.name}")
        raise typer.Exit(1)

    if output == "json":
        typer.echo(json_text(tensor.as_dict()))
    elif output == "table":
        table = Table(title="Tensor", show_header=False, box=None)
        table.add_column(style="bold cyan")
        table.add_column()
        values = tensor.as_dict()
        table.add_row("Name", tensor.name)
        table.add_row("Shape", format_shape(tensor.shape))
        table.add_row("Rank", str(tensor.rank))
        table.add_row("Dtype", tensor.dtype)
        table.add_row("Parameters", f"{tensor.numel:,}")
        table.add_row("Size", f"{tensor.size_bytes:,} bytes ({human_bytes(tensor.size_bytes)})")
        table.add_row("Shard", tensor.shard.name)
        table.add_row("Path", values["path"])
        table.add_row("Data offsets", str(values["data_offsets"]))
        console.print(table)
    else:
        raise_output_error(output, ("table", "json"))


@app.command("tree")
def tree_command(
    source: SourceArg,
    depth: Annotated[int | None, typer.Option(help="Maximum module depth to display")] = None,
    output: Annotated[str, typer.Option("--format", "-f", help="Output format: table or json")] = "table",
) -> None:
    """Display tensor names as a module hierarchy."""
    if depth is not None and depth < 1:
        err_console.print("[bold red]Error:[/] --depth must be at least 1")
        raise typer.Exit(2)
    model = get_model(source)
    nested = build_tree(model.tensors, depth)
    if output == "json":
        typer.echo(json_text(nested))
        return
    if output != "table":
        raise_output_error(output, ("table", "json"))

    rich_tree = Tree(f"[bold]{model.source.name}[/] ({len(model.tensors):,} tensors)")
    add_rich_branches(rich_tree, nested)
    console.print(rich_tree)


@app.command("verify")
def verify_command(
    source: SourceArg,
    output: Annotated[str, typer.Option("--format", "-f", help="Output format: table or json")] = "table",
) -> None:
    """Check headers, offsets, dtype sizes, shards, and the HF index."""
    model = get_model(source)
    issues = verify_model(model)
    errors = sum(issue.level == "error" for issue in issues)
    warnings = sum(issue.level == "warning" for issue in issues)
    result = {
        "ok": errors == 0,
        "errors": errors,
        "warnings": warnings,
        "tensor_count": len(model.tensors),
        "shard_count": len(model.shards),
        "issues": [issue.as_dict() for issue in issues],
    }
    if output == "json":
        typer.echo(json_text(result))
    elif output == "table":
        if not issues:
            console.print(
                f"[bold green]OK[/] — {len(model.tensors):,} tensors across {len(model.shards)} shard(s) passed verification."
            )
        else:
            table = Table(title=f"Verification: {errors} error(s), {warnings} warning(s)")
            table.add_column("Level")
            table.add_column("Message")
            for issue in issues:
                style = "red" if issue.level == "error" else "yellow"
                table.add_row(f"[{style}]{issue.level.upper()}[/]", issue.message)
            console.print(table)
    else:
        raise_output_error(output, ("table", "json"))
    if errors:
        raise typer.Exit(1)


@app.command("serve")
def serve_command(
    source: SourceArg,
    host: Annotated[str, typer.Option(help="Network interface to bind")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="TCP port to listen on", min=1, max=65535)] = 8000,
    open_browser: Annotated[
        bool,
        typer.Option("--open/--no-open", help="Open the viewer in the default browser"),
    ] = False,
) -> None:
    """Start a local web UI for a model."""
    model = get_model(source)
    from .server import create_app

    web_app = create_app(model=model)
    url_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{url_host}:{port}"
    console.print(f"[bold green]Safetensors Viewer[/] loaded {len(model.tensors):,} tensors.")
    console.print(f"Open [bold cyan]{url}[/] (API docs: {url}/api/docs)")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        console.print("[yellow]Warning:[/] the server is reachable from other machines on this network.")
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(0.5, webbrowser.open, args=(url,)).start()

    import uvicorn

    uvicorn.run(web_app, host=host, port=port, log_level="info")


def format_shape(shape: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(dim) for dim in shape) + "]"


def build_tree(tensors: list[TensorInfo], depth: int | None) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for tensor in sorted(tensors, key=lambda item: natural_key(item.name)):
        parts = tensor.name.split(".")
        if depth is not None and len(parts) > depth:
            parts = parts[:depth] + ["…"]
        node = root
        for part in parts:
            node = node.setdefault(part, {})
        node.setdefault("__tensors__", []).append(
            {"name": tensor.name, "shape": list(tensor.shape), "dtype": tensor.dtype}
        )
    return root


def add_rich_branches(parent: Tree, node: dict[str, Any]) -> None:
    keys = sorted((key for key in node if key != "__tensors__"), key=natural_key)
    for key in keys:
        child_data = node[key]
        tensors = child_data.get("__tensors__", [])
        label = key
        if tensors and not [child for child in child_data if child != "__tensors__"]:
            if key == "…":
                label = f"[dim]… ({len(tensors)} tensors)[/]"
            elif len(tensors) == 1:
                tensor = tensors[0]
                label = f"{key} [dim]{tensor['shape']} {tensor['dtype']}[/]"
            else:
                label = f"{key} [dim]({len(tensors)} tensors)[/]"
        branch = parent.add(label)
        add_rich_branches(branch, child_data)


def raise_output_error(output: str, allowed: tuple[str, ...]) -> None:
    err_console.print(f"[bold red]Error:[/] unsupported format {output!r}; choose {', '.join(allowed)}")
    raise typer.Exit(2)


if __name__ == "__main__":
    app()
