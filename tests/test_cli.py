from __future__ import annotations

import json

from typer.testing import CliRunner

from stview.cli import app
from test_model import create_model


runner = CliRunner()


def test_info_json(tmp_path) -> None:
    index = create_model(tmp_path)
    result = runner.invoke(app, ["info", str(index), "--format", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["tensor_count"] == 3
    assert payload["total_size_bytes"] == 30


def test_list_filter_json(tmp_path) -> None:
    index = create_model(tmp_path)
    result = runner.invoke(
        app,
        ["list", str(index), "--match", "blocks.*", "--ndim", "2", "--format", "json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [item["name"] for item in payload] == ["blocks.0.weight"]


def test_show_missing_returns_nonzero(tmp_path) -> None:
    index = create_model(tmp_path)
    result = runner.invoke(app, ["show", str(index), "missing"])
    assert result.exit_code == 1


def test_verify_valid_model(tmp_path) -> None:
    index = create_model(tmp_path)
    result = runner.invoke(app, ["verify", str(index), "--format", "json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["ok"] is True


def test_serve_starts_uvicorn(tmp_path, monkeypatch) -> None:
    index = create_model(tmp_path)
    called = {}

    def fake_run(web_app, **kwargs) -> None:
        called["app"] = web_app
        called.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    result = runner.invoke(app, ["serve", str(index), "--port", "8765", "--no-open"])
    assert result.exit_code == 0, result.output
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 8765
    assert called["app"].state.model.total_params == 9
