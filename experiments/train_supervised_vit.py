"""Finetune a supervised-pretrained ViT on OUHANDS using the existing loader utilities."""
from __future__ import annotations

import argparse
from pathlib import Path
from contextlib import nullcontext

import timm
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import ExperimentConfig, get_config
from train import create_datasets, create_dataloaders
from evaluate import classification_metrics, evaluate_model
from utils import select_device, set_seed


def build_supervised_vit(config: ExperimentConfig, device: torch.device) -> tuple[nn.Module, dict[str, object]]:
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model.reset_classifier(num_classes=config.model.num_classes)

    for param in model.parameters():
        param.requires_grad = False

    k = 3
    if hasattr(model, "blocks"):
        for block in model.blocks[-k:]:
            for param in block.parameters():
                param.requires_grad = True

    for param in model.get_classifier().parameters():
        param.requires_grad = True

    model.to(device)

    data_cfg = timm.data.resolve_model_data_config(model)
    transforms = {
        "train": timm.data.create_transform(**data_cfg, is_training=True),
        "eval": timm.data.create_transform(**data_cfg, is_training=False),
    }
    transforms.setdefault("train_pair", None)
    transforms.setdefault("eval_pair", None)
    return model, transforms


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
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


def train_one_epoch_supervised(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    max_grad_norm: float | None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    autocast_ctx = (
        (lambda: torch.amp.autocast("cuda")) if use_amp and device.type == "cuda" else nullcontext
    )

    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        labels = batch[1].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx():
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        preds = logits.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def run_training(args: argparse.Namespace) -> None:
    config: ExperimentConfig = get_config()
    device = select_device()
    print(f"Using device: {device}")

    set_seed(config.dataset.random_seed)

    model, transforms = build_supervised_vit(config, device)
    train_ds, val_ds, test_ds = create_datasets(config, transforms)
    train_loader, val_loader, test_loader = create_dataloaders(config, train_ds, val_ds, test_ds, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(param_groups(model, config.training.weight_decay), lr=config.training.learning_rate, betas=config.training.betas)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.scheduler_t_max)

    use_amp = device.type == "cuda" and config.training.use_mixed_precision
    scaler = GradScaler(enabled=use_amp)

    best_acc = 0.0
    best_state = None

    print("Training supervised ViT baseline")
    for epoch in range(1, config.training.epochs + 1):
        train_loss, train_acc = train_one_epoch_supervised(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_amp,
            config.training.max_grad_norm,
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
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        checkpoint_dir = Path(args.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / "vit_supervised_best.pt"
        torch.save(best_state, ckpt_path)
        print(f"Saved best checkpoint (val_acc={best_acc:.4f}) to {ckpt_path}")

    test_metrics = classification_metrics(model, test_loader, device, use_amp)
    print("Test metrics: " + " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune a supervised ViT baseline on OUHANDS")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Where to store the best checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
