from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


@dataclass
class BuildResult:
    model: nn.Module
    warnings: List[str]


class SequentialFromSpec(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x):  # noqa: ANN001
        for layer in self.layers:
            x = layer(x)
        return x


def build_torch_from_model_spec(model_spec: Dict[str, Any]) -> BuildResult:
    """
    Build a PyTorch model from the portable `model_spec.json`.
    """
    warnings: List[str] = []
    spec_layers = model_spec.get("layers", [])
    layers: List[nn.Module] = []

    for entry in spec_layers:
        kind = entry["kind"]
        params = entry.get("params", {})

        if kind == "linear":
            layers.append(
                nn.Linear(
                    in_features=int(params["in_features"]),
                    out_features=int(params["out_features"]),
                    bias=bool(params.get("bias", True)),
                )
            )
        elif kind == "conv2d":
            layers.append(
                nn.Conv2d(
                    in_channels=int(params["in_channels"]),
                    out_channels=int(params["out_channels"]),
                    kernel_size=tuple(params["kernel_size"]),
                    stride=tuple(params.get("stride", (1, 1))),
                    padding=tuple(params.get("padding", (0, 0))),
                    dilation=tuple(params.get("dilation", (1, 1))),
                    groups=int(params.get("groups", 1)),
                    bias=bool(params.get("bias", True)),
                )
            )
        elif kind == "batchnorm2d":
            layers.append(
                nn.BatchNorm2d(
                    num_features=int(params["num_features"]),
                    eps=float(params.get("eps", 1e-5)),
                    momentum=float(params.get("momentum", 0.1)),
                    affine=True,
                    track_running_stats=True,
                )
            )
        elif kind == "relu":
            layers.append(nn.ReLU(inplace=False))
        elif kind == "maxpool2d":
            layers.append(
                nn.MaxPool2d(
                    kernel_size=tuple(params["kernel_size"]),
                    stride=tuple(params.get("stride", params["kernel_size"])),
                    padding=tuple(params.get("padding", (0, 0))),
                )
            )
        elif kind == "flatten":
            layers.append(nn.Flatten(start_dim=1))
        elif kind == "global_avg_pool2d":
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            layers.append(nn.Flatten(start_dim=1))
        elif kind == "dropout":
            # Safe for inference (validator runs eval()).
            layers.append(nn.Dropout(p=float(params.get("p", 0.5))))
        else:
            warnings.append(f"Unsupported spec layer kind: {kind}")

    model = SequentialFromSpec(nn.ModuleList(layers))
    return BuildResult(model=model, warnings=warnings)


def infer_nchw_input(model_spec: Dict[str, Any]) -> Tuple[int, ...] | None:
    ish = model_spec.get("input_shape")
    if not ish:
        return None
    # Stored as TF-style NHWC (None,H,W,C) when available.
    try:
        _, h, w, c = ish
        return (1, int(c), int(h), int(w))
    except Exception:
        return None


def load_state_dict_bytes_into_model(model: nn.Module, state_dict_bytes: bytes) -> None:
    import io

    state = torch.load(io.BytesIO(state_dict_bytes), map_location="cpu")
    model.load_state_dict(state, strict=False)

