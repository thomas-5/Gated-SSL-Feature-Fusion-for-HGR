"""Retrain the DinoSwavFusionModel from a checkpoint."""
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

from config import ExperimentConfig, get_config
from train import create_datasets, create_dataloaders
from evaluate import classification_metrics, evaluate_model
from utils import select_device, set_seed
from fusion_model import DinoSwavFusionModel
from train_fusion import build_fusion_model, param_groups, train_one_epoch_fusion

def run_retraining(args: argparse.Namespace) -> None:
    config: ExperimentConfig = get_config()
    
    # Override epochs
    config.training.epochs = args.epochs
    
    # Force use_bounding_box to True for this model
    config.dataset.use_bounding_box = True
    
    # Enable segmentation if weight is positive
    if config.training.segmentation_kl_weight > 0:
        config.dataset.use_segmentation = True
    
    device = select_device()
    print(f"Using device: {device}")

    set_seed(config.dataset.random_seed)

    model, transforms = build_fusion_model(config, device)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle potential mismatch if model architecture changed
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print("Warning: Strict loading failed. Attempting to load compatible keys only...")
        # Filter out mismatched keys (e.g. if projection layers changed size)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        # Print ignored keys for debugging
        ignored_keys = [k for k in state_dict.keys() if k not in pretrained_dict]
        if ignored_keys:
            print(f"Ignored keys from checkpoint: {ignored_keys[:5]} ... ({len(ignored_keys)} total)")
            
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Loaded compatible weights.")
    
    train_ds, val_ds, test_ds = create_datasets(config, transforms)
    train_loader, val_loader, test_loader = create_dataloaders(config, train_ds, val_ds, test_ds, device)

    # Use Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(param_groups(model, config.training.weight_decay), lr=config.training.learning_rate, betas=config.training.betas)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.scheduler_t_max)

    use_amp = device.type == "cuda" and config.training.use_mixed_precision
    scaler = GradScaler(enabled=use_amp)

    best_acc = 0.0
    best_loss = float('inf')
    best_state = None
    
    # Initial evaluation
    print("Evaluating initial model performance...")
    val_loss, val_acc = evaluate_model(model, val_loader, criterion, device, use_amp)
    print(f"Initial Validation | loss={val_loss:.4f} acc={val_acc:.4f}")
    best_acc = val_acc
    best_loss = val_loss
    best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    print(f"Retraining DinoSwavFusionModel for {config.training.epochs} epochs")
    for epoch in range(1, config.training.epochs + 1):
        train_loss, train_acc = train_one_epoch_fusion(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_amp,
            config.training.max_grad_norm,
            use_segmentation=config.dataset.use_segmentation,
            kl_weight=config.training.segmentation_kl_weight,
        )
        # Evaluate
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

        save_checkpoint = False
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint = True
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint = True

        if save_checkpoint:
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            
            # Save immediately when we find a better model
            checkpoint_dir = Path(args.output_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = checkpoint_dir / "fusion_model_retrained_best.pt"
            torch.save(best_state, ckpt_path)
            print(f"  [New Best] Saved checkpoint (val_acc={best_acc:.4f}, val_loss={best_loss:.4f}) to {ckpt_path}")

    if best_state is not None:
        print(f"Training finished. Best validation accuracy: {best_acc:.4f}")

    # Test
    test_metrics = classification_metrics(model, test_loader, device, use_amp)
    print("Test metrics: " + " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain DinoSwavFusionModel from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file to load")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Where to store the best checkpoint")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to retrain")
    return parser.parse_args()


if __name__ == "__main__":
    run_retraining(parse_args())
