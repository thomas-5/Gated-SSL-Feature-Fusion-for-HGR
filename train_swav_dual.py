"""Train the SwavDualModel (Global + Local) on OUHANDS."""
from __future__ import annotations

import argparse
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import timm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from config import ExperimentConfig, get_config
from train import create_datasets, create_dataloaders, _unpack_batch
from utils import select_device, set_seed
from swav_dual_model import SwavDualModel

def build_swav_dual_model(config: ExperimentConfig, device: torch.device) -> tuple[nn.Module, dict[str, object]]:
    model = SwavDualModel(
        num_classes=config.model.num_classes,
        pretrained=True
    )
    model.to(device)
    
    # Transforms
    # We use standard ResNet transforms
    data_cfg = timm.data.resolve_data_config({}, model=timm.create_model('resnet50', pretrained=False))
    
    transforms = {
        "train": timm.data.create_transform(**data_cfg, is_training=True, auto_augment='rand-m9-mstd0.5-inc1'),
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

def evaluate_model_dual(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    """Compute average loss and accuracy for a dataloader, using bboxes."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    autocast_ctx = (
        (lambda: torch.amp.autocast("cuda")) if use_amp and device.type == "cuda" else nullcontext
    )

    with torch.no_grad():
        for batch in loader:
            images, labels, bboxes, masks, paths = _unpack_batch(
                batch, 
                use_bounding_box=True, 
                use_segmentation=False,
                return_paths=True
            )

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if bboxes is not None:
                if isinstance(bboxes, (list, tuple)):
                    bboxes = torch.stack(bboxes, dim=1)
                bboxes = bboxes.to(device, non_blocking=True)

            with autocast_ctx():
                logits = model(images, gt_bboxes=bboxes)
                loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = running_loss / total_samples
    avg_acc = running_correct / total_samples
    return avg_loss, avg_acc

def classification_metrics_dual(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    """Return accuracy, precision, recall and F1-score (macro averaged) using bboxes."""
    model.eval()
    all_preds, all_labels = [], []

    autocast_ctx = (
        (lambda: torch.amp.autocast("cuda")) if use_amp and device.type == "cuda" else nullcontext
    )

    with torch.no_grad():
        for batch in loader:
            images, labels, bboxes, masks, paths = _unpack_batch(
                batch, 
                use_bounding_box=True, 
                use_segmentation=False,
                return_paths=True
            )

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if bboxes is not None:
                if isinstance(bboxes, (list, tuple)):
                    bboxes = torch.stack(bboxes, dim=1)
                bboxes = bboxes.to(device, non_blocking=True)

            with autocast_ctx():
                logits = model(images, gt_bboxes=bboxes)

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

def train_one_epoch_dual(
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
        images, labels, bboxes, masks, paths = _unpack_batch(
            batch, 
            use_bounding_box=True, 
            use_segmentation=False,
            return_paths=True
        )
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if bboxes is not None:
            if isinstance(bboxes, (list, tuple)):
                bboxes = torch.stack(bboxes, dim=1)
            bboxes = bboxes.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx():
            try:
                logits = model(images, gt_bboxes=bboxes)
            except RuntimeError as e:
                if "Input and output sizes should be greater than 0" in str(e):
                    print("\n!!! CAUGHT EMPTY CROP ERROR !!!")
                    print("Batch paths:")
                    for i, p in enumerate(paths):
                        print(f"{i}: {p}")
                    print("Batch bboxes (x, y, w, h):")
                    print(bboxes)
                    raise e
                else:
                    raise e

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
    
    config.dataset.use_bounding_box = True
    config.dataset.use_segmentation = False
    
    device = select_device()
    print(f"Using device: {device}")

    set_seed(config.dataset.random_seed)

    model, transforms = build_swav_dual_model(config, device)
    train_ds, val_ds, test_ds = create_datasets(config, transforms)
    train_loader, val_loader, test_loader = create_dataloaders(config, train_ds, val_ds, test_ds, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(param_groups(model, config.training.weight_decay), lr=config.training.learning_rate, betas=config.training.betas)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.scheduler_t_max)

    use_amp = device.type == "cuda" and config.training.use_mixed_precision
    scaler = GradScaler(enabled=use_amp)

    best_acc = 0.0
    best_loss = float('inf')
    best_state = None

    print("Training SwavDualModel (Global + Local)")
    for epoch in range(1, config.training.epochs + 1):
        train_loss, train_acc = train_one_epoch_dual(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_amp,
            config.training.max_grad_norm,
        )
        
        val_loss, val_acc = evaluate_model_dual(
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

        save_checkpoint = False
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint = True
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint = True

        if save_checkpoint:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        checkpoint_dir = Path(args.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_dir / "swav_dual_best.pt"
        torch.save(best_state, ckpt_path)
        print(f"Saved best checkpoint (val_acc={best_acc:.4f}, val_loss={best_loss:.4f}) to {ckpt_path}")

    test_metrics = classification_metrics_dual(model, test_loader, device, use_amp)
    print("Test metrics: " + " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SwavDualModel on OUHANDS")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Where to store the best checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
