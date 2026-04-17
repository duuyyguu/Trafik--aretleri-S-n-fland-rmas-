from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as tvm


@dataclass(frozen=True)
class ModelSpec:
    name: str


def build_model(num_classes: int, spec: ModelSpec) -> nn.Module:
    name = spec.name.lower()
    if name in {"resnet18", "resnet-18"}:
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name in {"mobilenet_v3_small", "mobilenetv3small", "mobilenetv3-small"}:
        model = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    raise ValueError(f"Unknown model: {spec.name}")


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

