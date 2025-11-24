"""Training script for DINO fine-tuning on the OUHANDS dataset."""
from __future__ import annotations

import random
from contextlib import nullcontext
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from timm.layers import apply_rot_embed_cat

from ouhands_loader import OuhandsDS

from .config import ExperimentConfig, get_config
from .model import build_model, parameter_groups
from .evaluate import classification_metrics, evaluate_model


def select_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Ensure reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_datasets(
    config: ExperimentConfig,
    transforms: Dict[str, torch.nn.Module],
) -> Tuple[OuhandsDS, OuhandsDS, OuhandsDS]:
    """Instantiate train/validation/test datasets with shared configuration."""
    common_kwargs = {
        "root_dir": config.dataset.root_dir,
        "use_bounding_box": config.dataset.use_bounding_box,
        "crop_to_bbox": config.dataset.crop_to_bbox,
        "use_segmentation": config.dataset.use_segmentation,
    }

    train_ds = OuhandsDS(
        split="train",
        transform=transforms["train"],
        train_subset_ratio=config.dataset.train_subset_ratio,
        random_seed=config.dataset.random_seed,
        **common_kwargs,
    )

    val_ds = OuhandsDS(
        split="validation",
        transform=transforms["eval"],
        **common_kwargs,
    )

    test_ds = OuhandsDS(
        split="test",
        transform=transforms["eval"],
        **common_kwargs,
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


def _maybe_add_mask(attn: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Match timm's mask application semantics for attention tensors."""

    if attn_mask is None:
        return attn

    if attn_mask.dtype == torch.bool:
        return attn.masked_fill(~attn_mask, float("-inf"))

    return attn + attn_mask


def _attention_forward_with_weights(
    attn_module: nn.Module,
    x: torch.Tensor,
    rope: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom attention forward that also returns attention weights."""

    B, N, C = x.shape

    if getattr(attn_module, "qkv", None) is not None:
        if attn_module.q_bias is None:
            qkv = attn_module.qkv(x)
        else:
            qkv_bias = torch.cat((attn_module.q_bias, attn_module.k_bias, attn_module.v_bias))
            if attn_module.qkv_bias_separate:
                qkv = attn_module.qkv(x)
                qkv = qkv + qkv_bias
            else:
                qkv = F.linear(x, weight=attn_module.qkv.weight, bias=qkv_bias)

        qkv = qkv.reshape(B, N, 3, attn_module.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
    else:
        q = attn_module.q_proj(x).reshape(B, N, attn_module.num_heads, -1).transpose(1, 2)
        k = attn_module.k_proj(x).reshape(B, N, attn_module.num_heads, -1).transpose(1, 2)
        v = attn_module.v_proj(x).reshape(B, N, attn_module.num_heads, -1).transpose(1, 2)

    q, k = attn_module.q_norm(q), attn_module.k_norm(k)

    if rope is not None:
        num_prefix = attn_module.num_prefix_tokens
        half = getattr(attn_module, "rotate_half", False)
        q_suffix = apply_rot_embed_cat(q[:, :, num_prefix:, :], rope, half=half)
        k_suffix = apply_rot_embed_cat(k[:, :, num_prefix:, :], rope, half=half)
        q = torch.cat([q[:, :, :num_prefix, :], q_suffix.type_as(v)], dim=2)
        k = torch.cat([k[:, :, :num_prefix, :], k_suffix.type_as(v)], dim=2)

    q = q * attn_module.scale
    attn = q @ k.transpose(-2, -1)
    attn = _maybe_add_mask(attn, attn_mask)
    attn = attn.softmax(dim=-1)
    attn_dropped = attn_module.attn_drop(attn)

    x = attn_dropped @ v
    x = x.transpose(1, 2).reshape(B, N, C)
    x = attn_module.norm(x)
    x = attn_module.proj(x)
    x = attn_module.proj_drop(x)
    return x, attn


def _forward_block_with_attention(
    block: nn.Module, x: torch.Tensor, rope: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward a transformer block while capturing attention weights."""

    attn_out, attn_weights = _attention_forward_with_weights(block.attn, block.norm1(x), rope=rope)

    if block.gamma_1 is None:
        x = x + block.drop_path1(attn_out)
        x = x + block.drop_path2(block.mlp(block.norm2(x)))
    else:
        x = x + block.drop_path1(block.gamma_1 * attn_out)
        x = x + block.drop_path2(block.gamma_2 * block.mlp(block.norm2(x)))

    return x, attn_weights


def forward_with_attention(model: nn.Module, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass returning logits and final-block attention weights."""

    x = model.patch_embed(images)
    x, rot_pos_embed = model._pos_embed(x)
    x = model.norm_pre(x)

    attn_weights = None

    for idx, block in enumerate(model.blocks):
        if rot_pos_embed is None:
            rope = None
        elif getattr(model, "rope_mixed", False) and rot_pos_embed is not None:
            rope = rot_pos_embed[idx]
        else:
            rope = rot_pos_embed

        if idx == len(model.blocks) - 1:
            x, attn_weights = _forward_block_with_attention(block, x, rope)
        else:
            x = block(x, rope=rope)

    x = model.norm(x)
    logits = model.forward_head(x)
    return logits, attn_weights


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
        torch.cuda.amp.autocast if use_amp and device.type == "cuda" else nullcontext
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

    model, transforms = build_model(config, device)
    train_ds, val_ds, test_ds = create_datasets(config, transforms)
    train_loader, val_loader, test_loader = create_dataloaders(
        config, train_ds, val_ds, test_ds, device
    )

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

    best_acc = 0.0
    best_state: Dict[str, torch.Tensor] | None = None

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

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    checkpoint_path = config.checkpoint_path()
    if best_state is not None:
        torch.save(best_state, checkpoint_path)
        print(f"Saved best model (val_acc={best_acc:.4f}) to {checkpoint_path}")
        model.load_state_dict(best_state)
    else:
        print("Warning: No improvement observed during training: skipping checkpoint save.")

    metrics = classification_metrics(model, test_loader, device, use_amp)
    print(
        "Test metrics | "
        + " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    )


if __name__ == "__main__":
    main()
