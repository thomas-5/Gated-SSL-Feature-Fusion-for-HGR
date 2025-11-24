"""Evaluation utilities and script for DINO fine-tuning experiments."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from ouhands_loader import OuhandsDS

from .config import ExperimentConfig, get_config
from .model import build_model


def select_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        torch.cuda.amp.autocast if use_amp and device.type == "cuda" else nullcontext
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
        torch.cuda.amp.autocast if use_amp and device.type == "cuda" else nullcontext
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
        from .train import forward_with_attention  # Local import to avoid circular dependency

        _, attn_weights = forward_with_attention(model, image_tensor)

    attn_cls = attn_weights[:, :, 0, -num_patches:]
    attn_map = attn_cls.mean(dim=1)
    attn_map = attn_map.reshape(1, 1, grid_h, grid_w)
    attn_map = F.interpolate(attn_map, size=(img_size, img_size), mode="bicubic", align_corners=False)
    attn_map = attn_map.squeeze().cpu()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-6)

    return attn_map


def create_eval_dataloaders(
    config: ExperimentConfig,
    transforms: Dict[str, torch.nn.Module],
    device: torch.device,
) -> Dict[str, DataLoader]:
    """Build evaluation loaders for validation and test splits."""
    common_kwargs = {
        "root_dir": config.dataset.root_dir,
        "use_bounding_box": config.dataset.use_bounding_box,
        "crop_to_bbox": config.dataset.crop_to_bbox,
        "transform": transforms["eval"],
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

    checkpoint = checkpoint_path or config.checkpoint_path()
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


if __name__ == "__main__":
    run_evaluation()
