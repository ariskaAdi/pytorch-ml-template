import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.serve import create_app  # noqa: E402
from scripts.train import run  # noqa: E402


def test_predict_endpoint(make_cfg):
    cfg = make_cfg("train.precision=fp32", "train.max_steps=5")
    run(cfg)
    client = TestClient(create_app(f"{cfg.run_dir}/checkpoints/last.pt"))

    assert client.get("/health").json()["status"] == "ok"

    resp = client.post("/predict", json={"inputs": [[0.0] * 32, [1.0] * 32]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["labels"]) == 2
    assert len(body["probs"][0]) == 4
    assert abs(sum(body["probs"][0]) - 1.0) < 1e-4

    bad = client.post("/predict", json={"inputs": [[0.0] * 3]})
    assert bad.status_code == 422
