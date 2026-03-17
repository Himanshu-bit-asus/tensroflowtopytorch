from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.common.gcs import GcsPaths, upload_json
from services.common.tf_load import load_tf_model_from_gcs


class AnalyzeRequest(BaseModel):
    job_id: Optional[str] = None
    tf_model_uri: str
    artifacts_prefix_uri: str
    extra: Dict[str, Any] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    job_id: str
    blueprint_uri: str
    warnings: List[str] = Field(default_factory=list)


app = FastAPI(title="TF Model Analysis Agent", version="0.1")


def _keras_blueprint(model: tf.keras.Model) -> Dict[str, Any]:
    layers = []
    for layer in model.layers:
        layers.append(
            {
                "name": layer.name,
                "class_name": layer.__class__.__name__,
                "config": layer.get_config(),
                "weights": [
                    {
                        "name": w.name,
                        "shape": list(w.shape),
                        "dtype": w.dtype.name,
                    }
                    for w in layer.weights
                ],
            }
        )

    input_shapes = []
    output_shapes = []
    try:
        input_shapes = [list(s) if s is not None else None for s in model.input_shape]  # type: ignore[arg-type]
    except Exception:
        try:
            input_shapes = [list(model.input_shape)]  # type: ignore[list-item]
        except Exception:
            input_shapes = []

    try:
        output_shapes = [list(s) if s is not None else None for s in model.output_shape]  # type: ignore[arg-type]
    except Exception:
        try:
            output_shapes = [list(model.output_shape)]  # type: ignore[list-item]
        except Exception:
            output_shapes = []

    return {
        "kind": "keras",
        "name": model.name,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "layers": layers,
        "tf_version": tf.__version__,
    }


def _savedmodel_blueprint(obj: Any) -> Dict[str, Any]:
    sigs = getattr(obj, "signatures", {}) or {}
    sig_info = {}
    for k, fn in sigs.items():
        try:
            sig_info[k] = {
                "structured_input_signature": str(fn.structured_input_signature),
                "structured_outputs": str(fn.structured_outputs),
            }
        except Exception:
            sig_info[k] = {"error": "unable to introspect signature"}

    return {
        "kind": "saved_model",
        "signatures": sig_info,
        "tf_version": tf.__version__,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    job_id = req.job_id or uuid.uuid4().hex
    paths = GcsPaths(job_id=job_id, artifacts_prefix_uri=req.artifacts_prefix_uri)
    warnings: List[str] = []

    try:
        obj = load_tf_model_from_gcs(req.tf_model_uri)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load TF model: {e}") from e

    blueprint: Dict[str, Any]
    if isinstance(obj, tf.keras.Model):
        blueprint = _keras_blueprint(obj)
    else:
        # Some SavedModels can still be loaded via keras if they were exported that way;
        # we keep this path conservative and focus on signature-based blueprint.
        blueprint = _savedmodel_blueprint(obj)
        warnings.append(
            "Loaded a generic SavedModel (non-Keras). Conversion coverage may be limited; "
            "consider exporting as a Keras model (.keras/.h5) for best results."
        )

    blueprint.update(
        {
            "job_id": job_id,
            "tf_model_uri": req.tf_model_uri,
            "env": {
                "service": "analyzer",
                "region": os.environ.get("K_SERVICE", ""),
            },
            "extra": req.extra,
        }
    )

    blueprint_uri = upload_json(blueprint, paths.analysis_blueprint_uri())
    return AnalyzeResponse(job_id=job_id, blueprint_uri=blueprint_uri, warnings=warnings)

