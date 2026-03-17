from __future__ import annotations

import io
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from services.common.gcs import GcsPaths, download_json, upload_file, upload_json
from services.common.tf_load import load_tf_model_from_gcs
from services.converter.torch_builder import build_torch_from_model_spec


class ConvertRequest(BaseModel):
    job_id: Optional[str] = None
    tf_model_uri: str
    blueprint_uri: str
    artifacts_prefix_uri: str


class ConvertResponse(BaseModel):
    job_id: str
    model_spec_uri: str
    state_dict_uri: str
    warnings: List[str] = Field(default_factory=list)


app = FastAPI(title="TF→PT Conversion Agent", version="0.1")


def _tf_layer_class(layer: tf.keras.layers.Layer) -> str:
    return layer.__class__.__name__


def _conv2d_padding(tf_layer: tf.keras.layers.Conv2D) -> Tuple[int, int]:
    # TF "same" padding depends on input shape; we approximate with symmetric padding for stride=1.
    if str(tf_layer.padding).lower() == "same":
        kh, kw = tf_layer.kernel_size
        return (kh // 2, kw // 2)
    return (0, 0)


def _keras_to_model_spec(model: tf.keras.Model) -> Tuple[Dict[str, Any], List[str]]:
    warnings: List[str] = []
    spec_layers: List[Dict[str, Any]] = []

    input_shape = None
    try:
        input_shape = list(model.input_shape)  # type: ignore[list-item]
    except Exception:
        input_shape = None

    for layer in model.layers:
        cls = _tf_layer_class(layer)

        if isinstance(layer, tf.keras.layers.Dense):
            # kernel: (in, out)
            w = layer.get_weights()
            if not w:
                warnings.append(f"Dense layer '{layer.name}' has no weights.")
                continue
            in_features = int(w[0].shape[0])
            out_features = int(w[0].shape[1])
            spec_layers.append(
                {
                    "tf_name": layer.name,
                    "tf_class": cls,
                    "kind": "linear",
                    "params": {
                        "in_features": in_features,
                        "out_features": out_features,
                        "bias": layer.use_bias,
                    },
                }
            )
        elif isinstance(layer, tf.keras.layers.Conv2D):
            w = layer.get_weights()
            if not w:
                warnings.append(f"Conv2D layer '{layer.name}' has no weights.")
                continue
            # kernel: (kh, kw, in, out)
            kh, kw, in_ch, out_ch = w[0].shape
            padding_h, padding_w = _conv2d_padding(layer)
            spec_layers.append(
                {
                    "tf_name": layer.name,
                    "tf_class": cls,
                    "kind": "conv2d",
                    "params": {
                        "in_channels": int(in_ch),
                        "out_channels": int(out_ch),
                        "kernel_size": [int(kh), int(kw)],
                        "stride": [int(layer.strides[0]), int(layer.strides[1])],
                        "padding": [int(padding_h), int(padding_w)],
                        "dilation": [int(layer.dilation_rate[0]), int(layer.dilation_rate[1])],
                        "groups": 1,
                        "bias": layer.use_bias,
                    },
                }
            )
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            # gamma, beta, moving_mean, moving_variance
            w = layer.get_weights()
            if len(w) != 4:
                warnings.append(f"BatchNormalization '{layer.name}' weight format unexpected.")
                continue
            num_features = int(w[0].shape[0])
            spec_layers.append(
                {
                    "tf_name": layer.name,
                    "tf_class": cls,
                    "kind": "batchnorm2d",
                    "params": {
                        "num_features": num_features,
                        "eps": float(layer.epsilon),
                        "momentum": float(layer.momentum if layer.momentum is not None else 0.1),
                    },
                }
            )
        elif isinstance(layer, tf.keras.layers.ReLU):
            spec_layers.append({"tf_name": layer.name, "tf_class": cls, "kind": "relu", "params": {}})
        elif isinstance(layer, tf.keras.layers.Activation):
            act = layer.get_config().get("activation")
            if act == "relu":
                spec_layers.append({"tf_name": layer.name, "tf_class": cls, "kind": "relu", "params": {}})
            else:
                warnings.append(f"Unsupported Activation '{act}' in layer '{layer.name}'.")
        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            spec_layers.append(
                {
                    "tf_name": layer.name,
                    "tf_class": cls,
                    "kind": "maxpool2d",
                    "params": {
                        "kernel_size": [int(layer.pool_size[0]), int(layer.pool_size[1])],
                        "stride": [int(layer.strides[0]), int(layer.strides[1])]
                        if layer.strides is not None
                        else [int(layer.pool_size[0]), int(layer.pool_size[1])],
                        "padding": [0, 0],
                    },
                }
            )
        elif isinstance(layer, tf.keras.layers.Flatten):
            spec_layers.append({"tf_name": layer.name, "tf_class": cls, "kind": "flatten", "params": {}})
        elif isinstance(layer, tf.keras.layers.Dropout):
            spec_layers.append(
                {
                    "tf_name": layer.name,
                    "tf_class": cls,
                    "kind": "dropout",
                    "params": {"p": float(layer.rate)},
                }
            )
        elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            spec_layers.append(
                {"tf_name": layer.name, "tf_class": cls, "kind": "global_avg_pool2d", "params": {}}
            )
        elif isinstance(layer, tf.keras.layers.InputLayer):
            # Not a real op in PyTorch graph.
            continue
        else:
            warnings.append(f"Unsupported layer '{cls}' (name='{layer.name}').")

    model_spec = {
        "kind": "torch_sequential",
        "input_shape": input_shape,  # usually TF NHWC; validator handles transforms
        "layers": spec_layers,
        "source": {"tf_version": tf.__version__, "tf_model_name": model.name},
    }
    return model_spec, warnings


def _assign_weights_from_keras(model: tf.keras.Model, torch_model: nn.Module, model_spec: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    tf_by_name = {layer.name: layer for layer in model.layers}

    # Torch sequential stores layers in .layers (ModuleList).
    seq_layers: List[nn.Module] = list(getattr(torch_model, "layers"))

    spec_layers = model_spec.get("layers", [])
    if len(seq_layers) != len(spec_layers):
        warnings.append("Torch layer count differs from model_spec; weight loading may be partial.")

    for idx, spec in enumerate(spec_layers):
        tf_name = spec.get("tf_name")
        kind = spec.get("kind")
        if idx >= len(seq_layers):
            break
        pt_layer = seq_layers[idx]

        tf_layer = tf_by_name.get(tf_name) if tf_name else None
        if tf_layer is None:
            continue

        try:
            if kind == "linear" and isinstance(pt_layer, nn.Linear) and isinstance(tf_layer, tf.keras.layers.Dense):
                kernel, *rest = tf_layer.get_weights()
                bias = rest[0] if rest else None
                pt_layer.weight.data = torch.from_numpy(kernel.T.astype(np.float32))
                if pt_layer.bias is not None and bias is not None:
                    pt_layer.bias.data = torch.from_numpy(bias.astype(np.float32))
            elif kind == "conv2d" and isinstance(pt_layer, nn.Conv2d) and isinstance(tf_layer, tf.keras.layers.Conv2D):
                kernel, *rest = tf_layer.get_weights()
                bias = rest[0] if rest else None
                # TF: (kh, kw, in, out) -> PT: (out, in, kh, kw)
                pt_kernel = np.transpose(kernel, (3, 2, 0, 1)).astype(np.float32)
                pt_layer.weight.data = torch.from_numpy(pt_kernel)
                if pt_layer.bias is not None and bias is not None:
                    pt_layer.bias.data = torch.from_numpy(bias.astype(np.float32))
            elif kind == "batchnorm2d" and isinstance(pt_layer, nn.BatchNorm2d) and isinstance(
                tf_layer, tf.keras.layers.BatchNormalization
            ):
                gamma, beta, moving_mean, moving_var = tf_layer.get_weights()
                pt_layer.weight.data = torch.from_numpy(gamma.astype(np.float32))
                pt_layer.bias.data = torch.from_numpy(beta.astype(np.float32))
                pt_layer.running_mean.data = torch.from_numpy(moving_mean.astype(np.float32))
                pt_layer.running_var.data = torch.from_numpy(moving_var.astype(np.float32))
            else:
                # no weights or mismatched mapping
                pass
        except Exception as e:
            warnings.append(f"Failed loading weights for '{tf_name}' ({kind}): {e}")

    return warnings


@app.post("/convert", response_model=ConvertResponse)
def convert(req: ConvertRequest) -> ConvertResponse:
    job_id = req.job_id or uuid.uuid4().hex
    paths = GcsPaths(job_id=job_id, artifacts_prefix_uri=req.artifacts_prefix_uri)
    warnings: List[str] = []

    blueprint = download_json(req.blueprint_uri)
    if blueprint.get("kind") != "keras":
        warnings.append(
            f"Blueprint kind '{blueprint.get('kind')}' is not 'keras'. "
            "This converter currently focuses on Keras graphs."
        )

    try:
        obj = load_tf_model_from_gcs(req.tf_model_uri)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load TF model: {e}") from e

    if not isinstance(obj, tf.keras.Model):
        raise HTTPException(
            status_code=400,
            detail="Loaded a non-Keras SavedModel. Re-export as a Keras model (.keras/.h5) for this reference converter.",
        )

    model_spec, spec_warnings = _keras_to_model_spec(obj)
    warnings.extend(spec_warnings)

    build = build_torch_from_model_spec(model_spec)
    warnings.extend(build.warnings)
    torch_model = build.model
    torch_model.eval()

    warnings.extend(_assign_weights_from_keras(obj, torch_model, model_spec))

    model_spec_uri = upload_json(model_spec, paths.convert_model_spec_uri())

    # Serialize state_dict to bytes then upload.
    buf = io.BytesIO()
    torch.save(torch_model.state_dict(), buf)
    buf.seek(0)
    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(buf.read())

    state_dict_uri = upload_file(tmp_path, paths.convert_state_dict_uri(), content_type="application/octet-stream")
    return ConvertResponse(
        job_id=job_id,
        model_spec_uri=model_spec_uri,
        state_dict_uri=state_dict_uri,
        warnings=warnings,
    )

