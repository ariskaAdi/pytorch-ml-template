"""Minimal inference API.

CHECKPOINT_PATH=outputs/debug/checkpoints/best.pt uvicorn app.serve:app --port 8000
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"inputs": [[0.1, 0.2, ...]]}'
"""

import os

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mltemplate.inference.predict import load_model, predict


class PredictRequest(BaseModel):
    inputs: list[list[float]]


def create_app(checkpoint_path: str) -> FastAPI:
    model, cfg = load_model(checkpoint_path)
    in_dim = cfg.model.in_dim
    app = FastAPI(title="mltemplate inference")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "checkpoint": checkpoint_path}

    @app.post("/predict")
    def predict_endpoint(req: PredictRequest) -> dict:
        if not req.inputs or any(len(row) != in_dim for row in req.inputs):
            raise HTTPException(422, f"each input must have {in_dim} values")
        return predict(model, torch.tensor(req.inputs, dtype=torch.float32))

    return app


# Only build the module-level app when run as a server (keeps imports test-friendly).
if "CHECKPOINT_PATH" in os.environ:
    app = create_app(os.environ["CHECKPOINT_PATH"])
