"""Model construction utilities for DINO based experiments."""
from __future__ import annotations

from typing import Dict

import timm
import torch
import torch.nn as nn
from timm.data import create_transform, resolve_model_data_config

from .config import ExperimentConfig


def build_model(config: ExperimentConfig, device: torch.device) -> tuple[nn.Module, Dict[str, torch.nn.Module]]:
    """Instantiate the timm ViT model and prepare it for fine-tuning.

    Args:
        config: Experiment configuration instance.
        device: Torch device to place the model on.

    Returns:
        Tuple containing the prepared model and a dictionary with train/eval transforms.
    """
    model = timm.create_model(config.model.model_name, pretrained=True)
    model.reset_classifier(num_classes=config.model.num_classes)

    # Freeze everything first, then selectively unfreeze.
    for param in model.parameters():
        param.requires_grad = False

    total_blocks = len(getattr(model, "blocks", []))
    if total_blocks == 0:
        raise ValueError("Expected the model to expose transformer blocks for partial fine-tuning.")

    k = min(max(config.model.unfreeze_blocks, 0), total_blocks)
    if k > 0:
        for block in model.blocks[-k:]:
            for param in block.parameters():
                param.requires_grad = True

    # Classifier head always remains trainable.
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    model = model.to(device)

    data_config = resolve_model_data_config(model)
    transforms = {
        "train": create_transform(**data_config, is_training=True),
        "eval": create_transform(**data_config, is_training=False),
    }

    return model, transforms


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    """Create optimizer parameter groups that respect weight decay rules."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
