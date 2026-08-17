from __future__ import annotations

from fastapi.testclient import TestClient

from stview.model import load_model
from stview.server import create_app, module_summary
from test_model import create_model


def make_client(tmp_path) -> TestClient:
    model = load_model(create_model(tmp_path))
    return TestClient(create_app(model=model))


def test_web_assets(tmp_path) -> None:
    client = make_client(tmp_path)
    page = client.get("/")
    assert page.status_code == 200
    assert "Tensor index" in page.text
    assert client.get("/assets/app.css").headers["content-type"].startswith("text/css")
    assert client.get("/assets/app.js").status_code == 200


def test_info_api(tmp_path) -> None:
    response = make_client(tmp_path).get("/api/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["tensor_count"] == 3
    assert payload["total_params"] == 9
    assert payload["shard_count"] == 2


def test_tensor_api_filters_sorts_and_pages(tmp_path) -> None:
    client = make_client(tmp_path)
    response = client.get(
        "/api/tensors",
        params={"q": "blocks", "rank": 2, "sort": "size", "order": "desc", "limit": 1},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["name"] == "blocks.0.weight"

    response = client.get("/api/tensors", params={"dtype": "i16"})
    assert [item["name"] for item in response.json()["items"]] == ["head.weight"]


def test_tensor_detail_and_missing(tmp_path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/tensors/blocks.0.weight")
    assert response.status_code == 200
    assert response.json()["shape"] == [2, 2]
    assert client.get("/api/tensors/not-there").status_code == 404


def test_modules_and_verify(tmp_path) -> None:
    client = make_client(tmp_path)
    modules = client.get("/api/modules", params={"depth": 1}).json()
    assert [module["name"] for module in modules] == ["blocks", "head"]
    assert modules[0]["tensor_count"] == 2

    verification = client.post("/api/verify")
    assert verification.status_code == 200
    assert verification.json()["ok"] is True


def test_module_summary_uses_natural_order(tmp_path) -> None:
    model = load_model(create_model(tmp_path))
    summary = module_summary(model.tensors, depth=2)
    assert [item["name"] for item in summary] == ["blocks.0", "head.weight"]

