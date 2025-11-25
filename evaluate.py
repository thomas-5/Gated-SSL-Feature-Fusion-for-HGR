"""Evaluation utilities and script for DINO fine-tuning experiments."""
from __future__ import annotations

import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from ouhands_loader import OuhandsDS

from config import ExperimentConfig, get_config
from model import build_model
from utils import select_device
from vis_utils import save_attention_grid, visualize_attention_crop


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

    outputs_dir = Path("outputs")
    save_attention_grid(
        model,
        loaders["test"],
        device,
        output_dir=outputs_dir,
        seed=42,
    )

    test_dataset = loaders["test"].dataset
    if len(test_dataset) == 0:
        print("Test dataset is empty; skipping attention crop visualization.")
        return

    sample_idx = random.randrange(len(test_dataset))
    sample = test_dataset[sample_idx]
    if not isinstance(sample, tuple) or not sample:
        print("Unexpected sample format; skipping attention crop visualization.")
        return

    image_tensor = sample[0]
    label_value = sample[1] if len(sample) > 1 else None
    class_idx = None
    if label_value is not None:
        class_idx = int(label_value.item()) if isinstance(label_value, torch.Tensor) else int(label_value)

    class_name = str(class_idx) if class_idx is not None else "unknown"
    if class_idx is not None and hasattr(test_dataset, "get_class_name"):
        try:
            class_name = test_dataset.get_class_name(class_idx)
        except Exception:  # noqa: BLE001
            class_name = str(class_idx)

    sanitized_name = class_name.replace("/", "-").replace(" ", "_")
    crop_path = outputs_dir / f"sample_attention_crop_{sanitized_name}.png"
    visualize_attention_crop(
        model,
        image_tensor,
        device,
        save_path=crop_path,
    )
    print(f"Saved sample attention crop to {crop_path}")


if __name__ == "__main__":
    run_evaluation()
