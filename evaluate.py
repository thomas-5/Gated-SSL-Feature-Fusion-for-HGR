"""Evaluation utilities and script for DINO fine-tuning experiments."""
from __future__ import annotations

import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from ouhands_loader import OuhandsDS

from config import ExperimentConfig, get_config
from model import build_model
from utils import forward_with_attention, select_device


def _get_normalization_stats(model: nn.Module) -> tuple[Iterable[float], Iterable[float]]:
    """Return channel-wise mean/std used for preprocessing."""
    cfg = getattr(model, "pretrained_cfg", None) or {}
    mean = cfg.get("mean") or (0.485, 0.456, 0.406)
    std = cfg.get("std") or (0.229, 0.224, 0.225)
    return mean, std


def _denormalize_image(image: torch.Tensor, mean: Iterable[float], std: Iterable[float]) -> torch.Tensor:
    """Undo normalization for a single image tensor."""
    mean_tensor = torch.tensor(mean, dtype=image.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=image.dtype).view(-1, 1, 1)
    denorm = image.detach().cpu() * std_tensor + mean_tensor
    return denorm.clamp(0.0, 1.0)


def _load_mask_from_disk(
    dataset: OuhandsDS,
    sample_idx: int,
    image_tensor: torch.Tensor,
) -> torch.Tensor | None:
    """Load and resize segmentation mask for a dataset sample if available."""
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
    mask_tensor = F.interpolate(
        mask_tensor.unsqueeze(0), size=(target_h, target_w), mode="nearest"
    ).squeeze(0)

    return mask_tensor.squeeze(0)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Compute average loss and accuracy for a dataloader."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    autocast_ctx = (
        (lambda: torch.amp.autocast(device_type="cuda"))
        if use_amp and device.type == "cuda"
        else nullcontext
    )

    with torch.no_grad():
        for batch in loader:
            images = batch[0]
            labels = batch[1]

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast_ctx():
                logits = model(images)
                loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = running_loss / total_samples
    avg_acc = running_correct / total_samples
    return avg_loss, avg_acc


def classification_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    """Return accuracy, precision, recall and F1-score (macro averaged)."""
    model.eval()
    all_preds, all_labels = [], []

    autocast_ctx = (
        (lambda: torch.amp.autocast(device_type="cuda"))
        if use_amp and device.type == "cuda"
        else nullcontext
    )

    with torch.no_grad():
        for batch in loader:
            images = batch[0]
            labels = batch[1]

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast_ctx():
                logits = model(images)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = float((all_preds == all_labels).mean())
    precision = float(precision_score(all_labels, all_preds, average="macro", zero_division=0))
    recall = float(recall_score(all_labels, all_preds, average="macro", zero_division=0))
    f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def visualize_attention_map(
    image_tensor: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Return an attention heatmap resized to the input image shape."""
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    img_size = image_tensor.shape[-1]
    num_patches = model.patch_embed.num_patches
    grid_h, grid_w = model.patch_embed.grid_size

    with torch.no_grad():
        _, attn_weights = forward_with_attention(model, image_tensor)

    attn_cls = attn_weights[:, :, 0, -num_patches:]
    attn_map = attn_cls.mean(dim=1)
    attn_map = attn_map.reshape(1, 1, grid_h, grid_w)
    attn_map = F.interpolate(attn_map, size=(img_size, img_size), mode="bicubic", align_corners=False)
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
    seed: int = 23
) -> Path:
    """Randomly sample one image per class and save a 3x8 attention/GT grid."""
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

        mask_tensor = None
        if len(sample) >= 4 and getattr(dataset, "use_segmentation", False):
            mask_tensor = sample[3]

        sampled[label_idx] = (sample_idx, image_tensor, mask_tensor)
        if len(sampled) == len(target_classes):
            break

    if len(sampled) < len(target_classes):
        missing = sorted(set(target_classes) - set(sampled.keys()))
        raise RuntimeError(f"Could not sample all classes; missing {missing}")

    selected_entries: list[tuple[int, torch.Tensor, torch.Tensor | None]] = []
    selected_classes: list[int] = []
    for class_idx in target_classes:
        if class_idx not in sampled:
            continue
        selected_entries.append(sampled[class_idx])
        selected_classes.append(class_idx)
        if len(selected_entries) == num_images:
            break

    mean, std = _get_normalization_stats(model)
    model.eval()

    denorm_images: list[torch.Tensor] = []
    attention_maps: list[torch.Tensor] = []
    segmentation_masks: list[torch.Tensor | None] = []
    class_names: list[str] = []

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


def create_eval_dataloaders(
    config: ExperimentConfig,
    transforms: Dict[str, object],
    device: torch.device,
) -> Dict[str, DataLoader]:
    """Build evaluation loaders for validation and test splits."""
    eval_pair_transform = transforms.get("eval_pair")
    eval_transform = None if eval_pair_transform is not None else transforms.get("eval")

    common_kwargs = {
        "root_dir": config.dataset.root_dir,
        "use_bounding_box": config.dataset.use_bounding_box,
        "crop_to_bbox": config.dataset.crop_to_bbox,
        "use_segmentation": config.dataset.use_segmentation,
        "transform": eval_transform,
        "paired_transform": eval_pair_transform,
    }

    val_ds = OuhandsDS(split="validation", **common_kwargs)
    test_ds = OuhandsDS(split="test", **common_kwargs)

    pin_memory = device.type == "cuda"

    val_loader = DataLoader(
        val_ds,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=pin_memory,
    )

    return {"validation": val_loader, "test": test_loader}


def run_evaluation(config: ExperimentConfig | None = None, checkpoint_path: Path | None = None) -> None:
    """Evaluate a saved checkpoint on validation and test sets."""
    config = config or get_config()
    device = select_device()
    print(f"Using device: {device}")

    model, transforms = build_model(config, device)

    # Use last model
    checkpoint = checkpoint_path or config.checkpoint_path(mode='last')
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint}")

    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)

    loaders = create_eval_dataloaders(config, transforms, device)
    criterion = nn.CrossEntropyLoss()
    use_amp = config.training.use_mixed_precision and device.type == "cuda"

    val_loss, val_acc = evaluate_model(model, loaders["validation"], criterion, device, use_amp)
    print(f"Validation | loss={val_loss:.4f} acc={val_acc:.4f}")

    metrics = classification_metrics(model, loaders["test"], device, use_amp)
    print(
        "Test | " + " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    )

    save_attention_grid(
        model,
        loaders["test"],
        device,
        output_dir=Path("outputs"),
        seed=42,
    )


if __name__ == "__main__":
    run_evaluation()
