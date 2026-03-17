from __future__ import annotations

import io
import os
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.common.gcs import GcsPaths, download_bytes, download_json, upload_json
from services.common.tf_load import load_tf_model_from_gcs
from services.converter.torch_builder import build_torch_from_model_spec


class ValidationConfig(BaseModel):
    num_trials: int = 3
    atol: float = 1e-4
    rtol: float = 1e-3
    seed: int = 0


class ValidateRequest(BaseModel):
    job_id: Optional[str] = None
    tf_model_uri: str
    model_spec_uri: str
    state_dict_uri: str
    artifacts_prefix_uri: str
    validation: Optional[ValidationConfig] = None


class ValidateResponse(BaseModel):
    job_id: str
    validation_report_uri: str
    pass_: bool = Field(alias="pass")
    metrics: Dict[str, Any]


app = FastAPI(title="Validation Agent", version="0.1")


def _make_inputs_from_spec(model_spec: Dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """
    Returns (tf_input, layout) where layout is "nhwc" or "flat".
    """
    ish = model_spec.get("input_shape")
    if not ish or not isinstance(ish, list):
        # fallback: small flat tensor
        return rng.standard_normal((1, 8), dtype=np.float32), "flat"

    # Common Keras shapes:
    # - [None, features]
    # - [None, H, W, C]
    if len(ish) == 2:
        features = int(ish[1] or 8)
        return rng.standard_normal((1, features), dtype=np.float32), "flat"
    if len(ish) == 4:
        h = int(ish[1] or 32)
        w = int(ish[2] or 32)
        c = int(ish[3] or 3)
        return rng.standard_normal((1, h, w, c), dtype=np.float32), "nhwc"

    return rng.standard_normal((1, 8), dtype=np.float32), "flat"


def _torch_from_tf_input(x: np.ndarray, layout: str) -> torch.Tensor:
    t = torch.from_numpy(x.astype(np.float32))
    if layout == "nhwc":
        return t.permute(0, 3, 1, 2).contiguous()
    return t


def _compare(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return {"max_abs_diff": float("nan"), "mse": float("nan"), "cosine_similarity": float("nan")}
    max_abs = float(np.max(np.abs(a - b)))
    mse = float(np.mean((a - b) ** 2))
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    cos = float(np.dot(a, b) / denom)
    return {"max_abs_diff": max_abs, "mse": mse, "cosine_similarity": cos}


@app.post("/validate", response_model=ValidateResponse)
def validate(req: ValidateRequest) -> ValidateResponse:
    job_id = req.job_id or uuid.uuid4().hex
    paths = GcsPaths(job_id=job_id, artifacts_prefix_uri=req.artifacts_prefix_uri)
    cfg = req.validation or ValidationConfig()

    try:
        tf_obj = load_tf_model_from_gcs(req.tf_model_uri)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load TF model: {e}") from e
    if not isinstance(tf_obj, tf.keras.Model):
        raise HTTPException(status_code=400, detail="Validator expects a Keras model for this reference implementation.")
    tf_model: tf.keras.Model = tf_obj
    tf_model.trainable = False

    model_spec = download_json(req.model_spec_uri)
    build = build_torch_from_model_spec(model_spec)
    torch_model = build.model

    # Load state_dict
    state_bytes = download_bytes(req.state_dict_uri)
    state = torch.load(io.BytesIO(state_bytes), map_location="cpu")
    torch_model.load_state_dict(state, strict=False)
    torch_model.eval()

    rng = np.random.default_rng(cfg.seed)

    all_metrics: List[Dict[str, float]] = []
    pass_flags: List[bool] = []

    for _ in range(int(cfg.num_trials)):
        x_tf, layout = _make_inputs_from_spec(model_spec, rng)
        y_tf = tf_model(x_tf, training=False).numpy()

        x_pt = _torch_from_tf_input(x_tf, layout)
        with torch.no_grad():
            y_pt = torch_model(x_pt).cpu().numpy()

        m = _compare(y_tf, y_pt)
        all_metrics.append(m)
        pass_flags.append(bool(np.allclose(y_tf, y_pt, atol=cfg.atol, rtol=cfg.rtol)))

    # Aggregate conservatively (worst-case)
    max_abs = float(np.nanmax([m["max_abs_diff"] for m in all_metrics]))
    mse = float(np.nanmax([m["mse"] for m in all_metrics]))
    cos = float(np.nanmin([m["cosine_similarity"] for m in all_metrics]))
    passed = all(pass_flags)

    report = {
        "job_id": job_id,
        "pass": passed,
        "config": cfg.model_dump(),
        "per_trial": all_metrics,
        "aggregate": {"max_abs_diff": max_abs, "mse": mse, "cosine_similarity": cos},
        "env": {"service": "validator", "region": os.environ.get("K_SERVICE", "")},
    }

    report_uri = upload_json(report, paths.validate_report_uri())
    return ValidateResponse(job_id=job_id, validation_report_uri=report_uri, **{"pass": passed}, metrics=report["aggregate"])

