"""Model construction utilities for DINO based experiments."""
from __future__ import annotations

from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from timm.data import create_transform, resolve_model_data_config

from config import ExperimentConfig
from paired_transforms import SegmentationEvalTransform, SegmentationTrainTransform
from utils import forward_with_attention as backbone_forward_with_attention

class DinoSwavFusion(nn.Module):
    """Fuse DINO CLS embeddings with SwAV crops derived from attention maps."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        num_classes: int,
        swav_arch: str,
        swav_trainable: bool,
        attention_threshold: float,
        attention_margin: float,
        fusion_dim: int | None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.attention_threshold = attention_threshold
        self.attention_margin = attention_margin

        swav_model, swav_meta = self._load_swav_model(swav_arch)
        self.swav_dim = swav_model.fc.in_features
        swav_model.fc = nn.Identity()
        if not swav_trainable:
            for param in swav_model.parameters():
                param.requires_grad = False
        self.swav = swav_model
        self._swav_frozen = not swav_trainable
        if self._swav_frozen:
            self.swav.eval()

        dino_dim = getattr(backbone, "num_features", None) or backbone.head.in_features
        self.fusion_dim = fusion_dim or dino_dim

        self.align_swav = nn.Linear(self.swav_dim, dino_dim)
        self.mixing = nn.Linear(dino_dim, self.fusion_dim)
        self.gate = nn.Linear(dino_dim + self.swav_dim, self.fusion_dim)
        self.classifier = nn.Linear(self.fusion_dim, num_classes)

        dino_cfg = backbone.pretrained_cfg or {}
        dino_mean = torch.tensor(dino_cfg.get("mean", (0.485, 0.456, 0.406)), dtype=torch.float32)
        dino_std = torch.tensor(dino_cfg.get("std", (0.229, 0.224, 0.225)), dtype=torch.float32)
        self.register_buffer("dino_mean", dino_mean.view(1, -1, 1, 1), persistent=False)
        self.register_buffer("dino_std", dino_std.view(1, -1, 1, 1), persistent=False)

        swav_mean = torch.tensor(swav_meta["mean"], dtype=torch.float32)
        swav_std = torch.tensor(swav_meta["std"], dtype=torch.float32)
        self.register_buffer("swav_mean", swav_mean.view(1, -1, 1, 1), persistent=False)
        self.register_buffer("swav_std", swav_std.view(1, -1, 1, 1), persistent=False)
        self.swav_resolution = swav_meta["resolution"]

    @staticmethod
    def _load_swav_model(arch: str) -> Tuple[nn.Module, Dict[str, Tuple[float, ...]]]:
        if arch != "swav_resnet50":
            raise ValueError(
                f"Unsupported SwAV architecture '{arch}'. Currently only 'swav_resnet50' is supported."
            )

        try:
            swav_model = torch.hub.load("facebookresearch/swav:main", "resnet50")
        except Exception as exc:
            raise RuntimeError(
                "Failed to load SwAV weights via torch.hub. Ensure internet access or pre-download the weights."
            ) from exc

        meta = {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "resolution": (224, 224),
        }
        return swav_model, meta

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_attention(images)
        return logits

    def forward_with_attention(
        self,
        images: torch.Tensor,
        *,
        return_cls: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, attn_weights, cls_tokens = backbone_forward_with_attention(
            self.backbone, images, return_cls=True
        )
        boxes = self._attention_to_boxes(attn_weights, images.shape[-2:])
        swav_input = self._extract_swav_crops(images, boxes)
        swav_features = self._forward_swav(swav_input)
        logits, fused = self._fuse(cls_tokens, swav_features)
        if return_cls:
            return logits, attn_weights, fused
        return logits, attn_weights

    def _attention_to_boxes(self, attn_weights: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        grid_h, grid_w = self.backbone.patch_embed.grid_size
        num_patches = self.backbone.patch_embed.num_patches
        cls_attn = attn_weights[:, :, 0, -num_patches:]
        cls_attn = cls_attn.mean(dim=1)
        cls_attn = cls_attn.reshape(-1, grid_h, grid_w)

        boxes = []
        patch_h = image_size[0] / grid_h
        patch_w = image_size[1] / grid_w

        for attn_map in cls_attn:
            norm_map = attn_map / attn_map.max().clamp_min(1e-6)
            mask = norm_map >= self.attention_threshold
            if not mask.any():
                mask = norm_map >= norm_map.max()

            rows = torch.where(mask.any(dim=1))[0]
            cols = torch.where(mask.any(dim=0))[0]

            if len(rows) == 0 or len(cols) == 0:
                row_min, row_max = 0, grid_h - 1
                col_min, col_max = 0, grid_w - 1
            else:
                row_min, row_max = rows[0].item(), rows[-1].item()
                col_min, col_max = cols[0].item(), cols[-1].item()

            margin_y = max(0, int(round(self.attention_margin * grid_h)))
            margin_x = max(0, int(round(self.attention_margin * grid_w)))

            row_min = max(0, row_min - margin_y)
            row_max = min(grid_h - 1, row_max + margin_y)
            col_min = max(0, col_min - margin_x)
            col_max = min(grid_w - 1, col_max + margin_x)

            top = int(round(row_min * patch_h))
            left = int(round(col_min * patch_w))
            bottom = int(round((row_max + 1) * patch_h))
            right = int(round((col_max + 1) * patch_w))

            boxes.append((top, left, bottom, right))

        return torch.tensor(boxes, device=attn_weights.device, dtype=torch.long)

    def _extract_swav_crops(self, images: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        device = images.device
        unnormalized = images * self.dino_std.to(device) + self.dino_mean.to(device)
        crops = []
        height, width = images.shape[-2:]
        target_h, target_w = self.swav_resolution

        for img, (top, left, bottom, right) in zip(unnormalized, boxes):
            top_i = int(top.clamp(0, height - 1))
            left_i = int(left.clamp(0, width - 1))
            bottom_i = int(bottom.clamp(top_i + 1, height))
            right_i = int(right.clamp(left_i + 1, width))

            crop = img[:, top_i:bottom_i, left_i:right_i]
            if crop.numel() == 0:
                crop = img
            crop = F.interpolate(crop.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False)
            crops.append(crop)

        crops_tensor = torch.cat(crops, dim=0)
        crops_tensor = (crops_tensor - self.swav_mean.to(device)) / self.swav_std.to(device)
        return crops_tensor

    def _forward_swav(self, crops: torch.Tensor) -> torch.Tensor:
        if self._swav_frozen:
            with torch.no_grad():
                return self.swav(crops)
        return self.swav(crops)

    def _fuse(self, h_d: torch.Tensor, h_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        aligned_s = self.align_swav(h_s)
        proj_d = self.mixing(h_d)
        proj_s = self.mixing(aligned_s)
        gate_values = torch.sigmoid(self.gate(torch.cat([h_d, h_s], dim=-1)))
        fused = gate_values * proj_d + (1.0 - gate_values) * proj_s
        logits = self.classifier(fused)
        return logits, fused

    def train(self, mode: bool = True) -> DinoSwavFusion:  # type: ignore[override]
        super().train(mode)
        if self._swav_frozen:
            self.swav.eval()
        return self

def build_model(config: ExperimentConfig, device: torch.device) -> tuple[nn.Module, Dict[str, object]]:
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

    input_size = data_config.get("input_size")
    if not input_size:
        raise ValueError("Model data config missing input_size for segmentation transforms")
    size = input_size[-1]
    mean = data_config.get("mean", (0.485, 0.456, 0.406))
    std = data_config.get("std", (0.229, 0.224, 0.225))

    transforms["train_pair"] = SegmentationTrainTransform(size=size, mean=mean, std=std)
    transforms["eval_pair"] = SegmentationEvalTransform(size=size, mean=mean, std=std)

    model = DinoSwavFusion(
        backbone=model,
        num_classes=config.model.num_classes,
        swav_arch=config.model.swav_arch,
        swav_trainable=config.model.swav_trainable,
        attention_threshold=config.model.attention_threshold,
        attention_margin=config.model.attention_margin,
        fusion_dim=config.model.fusion_dim,
    ).to(device)

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
