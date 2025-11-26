"""Visualization utilities for attention-driven crops and SwAV fusion."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ouhands_loader import OuhandsDS
from utils import model_forward_with_attention


@dataclass
class AttentionCropResult:
    image: torch.Tensor
    attention_map: torch.Tensor
    bbox: Tuple[int, int, int, int]
    crop: torch.Tensor
    crop_resized: torch.Tensor


def _get_normalization_stats(model: nn.Module) -> tuple[Iterable[float], Iterable[float]]:
    vit_model = getattr(model, "backbone", model)
    cfg = getattr(vit_model, "pretrained_cfg", None) or {}
    mean = cfg.get("mean") or (0.485, 0.456, 0.406)
    std = cfg.get("std") or (0.229, 0.224, 0.225)
    return mean, std


def _denormalize_image(image: torch.Tensor, mean: Iterable[float], std: Iterable[float]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=image.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image.dtype).view(-1, 1, 1)
    denorm = image.detach().cpu() * std_tensor + mean_tensor
    return denorm.clamp(0.0, 1.0)


def _load_mask_from_disk(dataset: OuhandsDS, sample_idx: int, image_tensor: torch.Tensor) -> torch.Tensor | None:
    if not hasattr(dataset, "samples"):
        return None

    try:
        image_path = dataset.samples[sample_idx][0]
    except (IndexError, TypeError):
        return None

    if not hasattr(dataset, "_load_segmentation_mask"):
        return None

    mask_pil = dataset._load_segmentation_mask(image_path.name)  # type: ignore[attr-defined]
    if mask_pil is None:
        return None

    mask_array = np.array(mask_pil, dtype=np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_array).float()
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0)

    target_h, target_w = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
    mask_tensor = F.interpolate(mask_tensor.unsqueeze(0), size=(target_h, target_w), mode="nearest").squeeze(0)

    return mask_tensor.squeeze(0)


def _normalize_to_numpy(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[0] in {1, 3}:
        tensor = tensor.detach().cpu()
        return tensor
    return tensor.detach().cpu()


def _plot_bbox(ax: plt.Axes, top: int, left: int, bottom: int, right: int, **kwargs: Any) -> None:
    rect = plt.Rectangle((left, top), right - left, bottom - top, fill=False, **kwargs)
    ax.add_patch(rect)


def visualize_attention_map(
    image_tensor: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    vit_model = getattr(model, "backbone", model)
    num_patches = vit_model.patch_embed.num_patches
    grid_h, grid_w = vit_model.patch_embed.grid_size

    with torch.no_grad():
        _, attn_weights = model_forward_with_attention(model, image_tensor)

    attn_cls = attn_weights[:, :, 0, -num_patches:]
    attn_map = attn_cls.mean(dim=1)
    attn_map = attn_map.reshape(1, 1, grid_h, grid_w)
    attn_map = F.interpolate(attn_map, size=(image_tensor.shape[-1], image_tensor.shape[-1]), mode="bicubic", align_corners=False)
    attn_map = attn_map.squeeze().cpu()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)
    return attn_map


def save_attention_grid(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    filename: str = "test_attention_grid.png",
    num_images: int = 10,
    seed: int = 23,
) -> Path:
    if num_images <= 0:
        raise ValueError("num_images must be positive")

    dataset = loader.dataset
    if not hasattr(dataset, "class_to_idx"):
        raise RuntimeError("Dataset must define class_to_idx for class sampling")

    class_to_idx: Dict[str, int] = dataset.class_to_idx  # type: ignore[attr-defined]
    target_classes = sorted(class_to_idx.values())

    indices = list(range(len(dataset)))
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(indices)
    else:
        random.shuffle(indices)

    sampled: Dict[int, tuple[int, torch.Tensor, torch.Tensor | None]] = {}
    for sample_idx in indices:
        sample = dataset[sample_idx]
        if not isinstance(sample, tuple) or len(sample) < 2:
            continue

        image_tensor = sample[0]
        label_tensor = sample[1]
        label_idx = int(label_tensor.item()) if isinstance(label_tensor, torch.Tensor) else int(label_tensor)
        if label_idx in sampled:
            continue

        mask_tensor = sample[2] if len(sample) >= 3 else None

        sampled[label_idx] = (sample_idx, image_tensor, mask_tensor)
        if len(sampled) == len(target_classes):
            break

    if len(sampled) < len(target_classes):
        missing = sorted(set(target_classes) - set(sampled.keys()))
        raise RuntimeError(f"Could not sample all classes; missing {missing}")

    selected_entries: List[tuple[int, torch.Tensor, torch.Tensor | None]] = []
    selected_classes: List[int] = []
    for class_idx in target_classes:
        if class_idx not in sampled:
            continue
        selected_entries.append(sampled[class_idx])
        selected_classes.append(class_idx)
        if len(selected_entries) == num_images:
            break

    mean, std = _get_normalization_stats(model)
    model.eval()

    denorm_images: List[torch.Tensor] = []
    attention_maps: List[torch.Tensor] = []
    segmentation_masks: List[torch.Tensor | None] = []
    class_names: List[str] = []

    with torch.no_grad():
        for class_idx, (sample_idx, image_tensor, mask_tensor) in zip(selected_classes, selected_entries):
            normalized_img = image_tensor.detach().cpu()
            denorm_images.append(_denormalize_image(normalized_img, mean, std))
            attn_map = visualize_attention_map(normalized_img, model, device)
            attention_maps.append(attn_map.cpu())

            seg_tensor = None
            if mask_tensor is not None:
                seg_tensor = mask_tensor.detach().cpu().float()
                if seg_tensor.dim() == 3 and seg_tensor.shape[0] == 1:
                    seg_tensor = seg_tensor.squeeze(0)
            else:
                seg_tensor = _load_mask_from_disk(dataset, sample_idx, normalized_img)

            segmentation_masks.append(seg_tensor)

            if hasattr(dataset, "get_class_name"):
                class_names.append(dataset.get_class_name(class_idx))
            else:
                class_names.append(str(class_idx))

    rows, cols = 3, 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes_grid = np.array(axes).reshape(rows, cols)

    for row in range(rows):
        for col in range(cols):
            axes_grid[row, col].axis("off")

    pairs_per_row = cols // 2
    for idx, (image, attn_map, mask, title) in enumerate(
        zip(denorm_images, attention_maps, segmentation_masks, class_names)
    ):
        row = idx // pairs_per_row
        col_pair = idx % pairs_per_row
        attn_ax = axes_grid[row, col_pair * 2]
        mask_ax = axes_grid[row, col_pair * 2 + 1]

        img_np = image.permute(1, 2, 0).numpy()
        attn_np = attn_map.numpy()

        attn_ax.imshow(img_np)
        attn_ax.imshow(attn_np, cmap="viridis", alpha=0.5)
        attn_ax.set_title(title)
        attn_ax.axis("off")

        if mask is not None:
            mask_ax.imshow(mask.numpy(), cmap="gray")
            mask_ax.set_title(f"{title} GT")
        mask_ax.axis("off")

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved attention map grid to {output_path}")
    return output_path


def visualize_attention_crop(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    save_path: Optional[Path] = None,
) -> AttentionCropResult:
    if image.dim() == 4 and image.size(0) == 1:
        image = image.squeeze(0)
    if image.dim() != 3:
        raise ValueError(
            "visualize_attention_crop expects a single image tensor in CHW format"
        )

    model.eval()
    with torch.no_grad():
        logits, attn_weights = model_forward_with_attention(model, image.unsqueeze(0).to(device))

    if hasattr(model, "backbone"):
        vit = model.backbone
    else:
        vit = model

    num_patches = vit.patch_embed.num_patches
    grid_h, grid_w = vit.patch_embed.grid_size
    attn_cls = attn_weights[:, :, 0, -num_patches:]
    attn_map = attn_cls.mean(dim=1).reshape(grid_h, grid_w)

    norm_map = attn_map / attn_map.max().clamp_min(1e-6)
    threshold = getattr(model, "attention_threshold", 0.6)
    high_attention_mask = norm_map >= threshold
    rows = torch.where(high_attention_mask.any(dim=1))[0]
    cols = torch.where(high_attention_mask.any(dim=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        row_min, row_max = 0, grid_h - 1
        col_min, col_max = 0, grid_w - 1
    else:
        row_min, row_max = rows[0].item(), rows[-1].item()
        col_min, col_max = cols[0].item(), cols[-1].item()

    margin = getattr(model, "attention_margin", 0.15)
    margin_y = max(1, int(round(margin * grid_h)))
    margin_x = max(1, int(round(margin * grid_w)))
    row_min = max(0, row_min - margin_y)
    row_max = min(grid_h - 1, row_max + margin_y)
    col_min = max(0, col_min - margin_x)
    col_max = min(grid_w - 1, col_max + margin_x)

    img_h, img_w = image.shape[-2:]
    patch_h = img_h / grid_h
    patch_w = img_w / grid_w
    top = int(round(row_min * patch_h))
    left = int(round(col_min * patch_w))
    bottom = int(round((row_max + 1) * patch_h))
    right = int(round((col_max + 1) * patch_w))

    bbox = (top, left, bottom, right)

    if hasattr(model, "dino_mean") and hasattr(model, "dino_std"):
        mean = model.dino_mean.to(image.device)
        std = model.dino_std.to(image.device)
        if mean.dim() == image.dim() + 1:
            mean = mean.squeeze(0)
        if std.dim() == image.dim() + 1:
            std = std.squeeze(0)
        unnormalized = image * std + mean
    else:
        unnormalized = image
    crop = unnormalized[:, top:bottom, left:right]
    if crop.numel() == 0:
        crop = unnormalized

    resolution = getattr(model, "swav_resolution", (224, 224))
    if isinstance(resolution, int):
        target_h = target_w = resolution
    else:
        target_h, target_w = resolution

    crop_to_resize = crop
    if crop_to_resize.dim() == 2:
        crop_to_resize = crop_to_resize.unsqueeze(0)
    if crop_to_resize.dim() == 3:
        crop_to_resize = crop_to_resize.unsqueeze(0)
    crop_to_resize = crop_to_resize.contiguous()

    if crop_to_resize.dim() != 4:
        raise ValueError("Unexpected crop dimensionality while preparing SwAV resize")

    crop_resized = F.interpolate(
        crop_to_resize,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    img_np = _normalize_to_numpy(unnormalized)
    attn_np = F.interpolate(attn_map.unsqueeze(0).unsqueeze(0), size=image.shape[-2:], mode="bilinear", align_corners=False).squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np.permute(1, 2, 0))
    _plot_bbox(axes[0], top, left, bottom, right, edgecolor="lime", linewidth=2)
    axes[0].set_title("RGB + MBR")
    axes[0].axis("off")

    axes[1].imshow(attn_np.cpu(), cmap="viridis")
    _plot_bbox(axes[1], top, left, bottom, right, edgecolor="white", linewidth=2)
    axes[1].set_title("Attention heatmap")
    axes[1].axis("off")

    axes[2].imshow(crop_resized.permute(1, 2, 0))
    axes[2].set_title("SwAV crop (resized)")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.close(fig)

    return AttentionCropResult(
        image=img_np,
        attention_map=attn_np,
        bbox=bbox,
        crop=crop.detach().cpu(),
        crop_resized=crop_resized.detach().cpu(),
    )
