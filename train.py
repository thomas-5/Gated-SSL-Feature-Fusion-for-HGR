"""Training script for DINO fine-tuning on the OUHANDS dataset."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ouhands_loader import OuhandsDS

from config import ExperimentConfig, get_config
from model import build_model, parameter_groups
from evaluate import classification_metrics, evaluate_model
from utils import forward_with_attention, select_device, set_seed


def create_datasets(
    config: ExperimentConfig,
    transforms: Dict[str, object],
) -> Tuple[OuhandsDS, OuhandsDS, OuhandsDS]:
    """Instantiate train/validation/test datasets with shared configuration."""
    train_pair_transform = transforms.get("train_pair")
    eval_pair_transform = transforms.get("eval_pair")

    train_transform = None if train_pair_transform is not None else transforms.get("train")
    eval_transform = None if eval_pair_transform is not None else transforms.get("eval")

    common_kwargs = {
        "root_dir": config.dataset.root_dir,
        "use_bounding_box": config.dataset.use_bounding_box,
        "crop_to_bbox": config.dataset.crop_to_bbox,
        "use_segmentation": config.dataset.use_segmentation,
    }

    train_kwargs = {
        **common_kwargs,
        "transform": train_transform,
        "paired_transform": train_pair_transform,
    }
    eval_kwargs = {
        **common_kwargs,
        "transform": eval_transform,
        "paired_transform": eval_pair_transform,
    }

    train_ds = OuhandsDS(
        split="train",
        **train_kwargs,
        train_subset_ratio=config.dataset.train_subset_ratio,
        random_seed=config.dataset.random_seed,
    )

    val_ds = OuhandsDS(
        split="validation",
        **eval_kwargs,
    )

    test_ds = OuhandsDS(
        split="test",
        **eval_kwargs,
    )

    return train_ds, val_ds, test_ds

def create_dataloaders(
    config: ExperimentConfig,
    train_ds: OuhandsDS,
    val_ds: OuhandsDS,
    test_ds: OuhandsDS,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch dataloaders for each dataset split."""
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.dataset.batch_size,
        shuffle=False,
        num_workers=config.dataset.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

def _unpack_batch(
    batch: Tuple,
    use_bounding_box: bool,
    use_segmentation: bool,
    return_paths: bool,
):
    """Split a variable-length batch tuple into its constituent parts."""

    idx = 0
    images = batch[idx]
    idx += 1

    labels = batch[idx]
    idx += 1

    bbox = None
    if use_bounding_box:
        bbox = batch[idx]
        idx += 1

    mask = None
    if use_segmentation:
        mask = batch[idx]
        idx += 1

    paths = None
    if return_paths:
        paths = batch[idx]

    return images, labels, bbox, mask, paths

def attention_mask_kl_loss(
    attn_weights: torch.Tensor,
    masks: torch.Tensor,
    model: nn.Module,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute KL divergence between CLS attention maps and segmentation masks."""

    num_patches = model.patch_embed.num_patches
    grid_h, grid_w = model.patch_embed.grid_size

    cls_attn = attn_weights[:, :, 0, -num_patches:]
    cls_attn = cls_attn.mean(dim=1)
    cls_attn = cls_attn / cls_attn.sum(dim=1, keepdim=True).clamp_min(eps)

    masks = F.interpolate(masks, size=(grid_h, grid_w), mode="nearest")
    mask_flat = masks.flatten(1)
    mask_sum = mask_flat.sum(dim=1, keepdim=True)
    valid = mask_sum.squeeze(1) > eps

    if not valid.any():
        return torch.zeros((), device=masks.device, dtype=masks.dtype)

    cls_attn = cls_attn[valid]
    mask_flat = mask_flat[valid]
    mask_sum = mask_sum[valid]

    mask_probs = mask_flat / mask_sum
    mask_probs = mask_probs.clamp_min(eps)
    cls_attn = cls_attn.clamp_min(eps)

    kl = mask_probs * (mask_probs.log() - cls_attn.log())
    kl = kl.sum(dim=1)
    return kl.mean()

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    use_amp: bool,
    max_grad_norm: float | None,
    use_bounding_box: bool,
    use_segmentation: bool,
    kl_weight: float,
) -> Tuple[float, float]:
    """Run a single training epoch."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    running_seg_loss = 0.0
    seg_samples = 0

    autocast_ctx = (
        (lambda: torch.amp.autocast(device_type="cuda"))
        if use_amp and device.type == "cuda"
        else nullcontext
    )

    progress = tqdm(loader, desc="Train", leave=False)
    for batch in progress:
        images, labels, _, masks, _ = _unpack_batch(
            batch,
            use_bounding_box=use_bounding_box,
            use_segmentation=use_segmentation,
            return_paths=False,
        )

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True) if masks is not None else None

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx():
            logits, attn_weights = forward_with_attention(model, images)
            loss = criterion(logits, labels)

            seg_loss = None
            if use_segmentation and masks is not None and kl_weight > 0:
                seg_loss = attention_mask_kl_loss(attn_weights, masks, model)
                loss = loss + kl_weight * seg_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        preds = logits.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

        if seg_loss is not None:
            running_seg_loss += seg_loss.item() * images.size(0)
            seg_samples += images.size(0)

        progress.set_postfix(
            loss=running_loss / total_samples,
            acc=running_correct / total_samples,
            seg_loss=(running_seg_loss / seg_samples) if seg_samples > 0 else 0.0,
        )

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def main(config: ExperimentConfig | None = None) -> None:
    """Entry point for training the DINO fine-tuning experiment."""
    config = config or get_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.dataset.random_seed)

    device = select_device()
    print(f"Using device: {device}")

    # Load model and datasets
    model, transforms = build_model(config, device)
    train_ds, val_ds, test_ds = create_datasets(config, transforms)
    train_loader, val_loader, test_loader = create_dataloaders(
        config, train_ds, val_ds, test_ds, device
    )

    # Prepare training components
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        parameter_groups(model, config.training.weight_decay),
        lr=config.training.learning_rate,
        betas=config.training.betas,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.training.scheduler_t_max
    )

    use_amp = config.training.use_mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Training loop
    best_acc = 0.0
    best_state: Dict[str, torch.Tensor] | None = None
    last_state: Dict[str, torch.Tensor] | None = None
    for epoch in range(1, config.training.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_amp,
            config.training.max_grad_norm,
            config.dataset.use_bounding_box,
            config.dataset.use_segmentation,
            config.training.segmentation_kl_weight,
        )
        val_loss, val_acc = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            use_amp,
        )

        scheduler.step()

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        last_state = {k: v.cpu() for k, v in model.state_dict().items()}
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    
    checkpoint_path = config.checkpoint_path(mode='last')
    torch.save(last_state, checkpoint_path)
    if best_state is not None:
        checkpoint_path = config.checkpoint_path(mode='best')
        torch.save(best_state, checkpoint_path)
        print(f"Saved best model (val_acc={best_acc:.4f}) to {checkpoint_path}")
    else:
        print("Warning: No improvement observed during training: skipping checkpoint save.")

    # Use last model for test evaluation TODO: change to best model if desired
    metrics = classification_metrics(model, test_loader, device, use_amp)
    print(
        "Test metrics | "
        + " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    )


if __name__ == "__main__":
    main()
