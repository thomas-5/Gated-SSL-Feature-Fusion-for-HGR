"""
Train the DinoSwavFusionModel with FROZEN backbones.
Loads pre-trained weights for DINO and SwAV, freezes them, and trains only the fusion layers.
"""
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

def load_pretrained_backbones(
    model: DinoSwavFusionModel, 
    dino_path: str, 
    swav_path: str, 
    device: torch.device
) -> None:
    """
    Load pre-trained weights into the fusion model's backbones.
    """
    print(f"Loading DINO weights from {dino_path}...")
    dino_ckpt = torch.load(dino_path, map_location=device)
    
    # The DINO checkpoint is the state_dict of the timm model (from train.py)
    # We load it into model.dino
    # We use strict=False because the checkpoint has a classification head (num_classes=10)
    # but model.dino might have a different head or we don't care about it.
    # We only care about the backbone.
    missing, unexpected = model.dino.load_state_dict(dino_ckpt, strict=False)
    print(f"DINO Load Results - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    # Expected missing: maybe nothing if shapes match, or head if we ignore it.
    # Expected unexpected: head weights if shapes mismatch.
    
    print(f"Loading SwAV weights from {swav_path}...")
    swav_ckpt = torch.load(swav_path, map_location=device)
    
    # The SwAV checkpoint is from SwavDualModel, which has 'backbone' and 'classifier'.
    # We want 'backbone.*' weights to go into model.swav
    swav_backbone_dict = {}
    for k, v in swav_ckpt.items():
        if k.startswith("backbone."):
            # Remove "backbone." prefix
            new_k = k[9:]
            swav_backbone_dict[new_k] = v
            
    # Load into model.swav
    # strict=False just in case, but should match mostly.
    missing, unexpected = model.swav.load_state_dict(swav_backbone_dict, strict=False)
    print(f"SwAV Load Results - Missing: {len(missing)}, Unexpected: {len(unexpected)}")

def run_frozen_training(args: argparse.Namespace) -> None:
    config: ExperimentConfig = get_config()
    
    # Override epochs if provided
    if args.epochs:
        config.training.epochs = args.epochs
        
    config.dataset.use_bounding_box = True
    
    # Enable segmentation if weight is positive
    if config.training.segmentation_kl_weight > 0:
        config.dataset.use_segmentation = True
    
    device = select_device()
    print(f"Using device: {device}")

    set_seed(config.dataset.random_seed)

    # Build model
    model, transforms = build_fusion_model(config, device)
    
    # Load Pretrained Weights
    load_pretrained_backbones(model, args.dino_checkpoint, args.swav_checkpoint, device)
    
    # Freeze Backbones
    print("Freezing DINO and SwAV backbones...")
    for param in model.dino.parameters():
        param.requires_grad = False
    for param in model.swav.parameters():
        param.requires_grad = False
        
    # Verify trainable parameters
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)}")
    print("Trainable layers:", [n for n in trainable_params if "weight" in n])
    
    # Create Datasets & Loaders
    train_ds, val_ds, test_ds = create_datasets(config, transforms)
    train_loader, val_loader, test_loader = create_dataloaders(config, train_ds, val_ds, test_ds, device)

    # Optimizer & Loss
    # Only pass trainable parameters to optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                      lr=config.training.learning_rate, 
                      weight_decay=config.training.weight_decay)
                      
    scheduler = CosineAnnealingLR(optimizer, T_max=config.training.scheduler_t_max)

    use_amp = device.type == "cuda" and config.training.use_mixed_precision
    scaler = GradScaler(enabled=use_amp)

    best_acc = 0.0
    best_loss = float('inf')
    best_state = None

    print("Starting training with FROZEN backbones...")
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
            
            # Save best
            checkpoint_dir = Path(args.output_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = checkpoint_dir / "fusion_frozen_best.pt"
            torch.save(best_state, ckpt_path)
            print(f"  [New Best] Saved checkpoint (val_acc={best_acc:.4f}, val_loss={best_loss:.4f}) to {ckpt_path}")

    if best_state is not None:
        print(f"Training finished. Best validation accuracy: {best_acc:.4f}")

    # Test
    test_metrics = classification_metrics(model, test_loader, device, use_amp)
    print("Test metrics: " + " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Fusion Model with Frozen Backbones")
    parser.add_argument("--dino-checkpoint", type=str, required=True, help="Path to DINO checkpoint")
    parser.add_argument("--swav-checkpoint", type=str, required=True, help="Path to SwAV checkpoint")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Where to store the best checkpoint")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    return parser.parse_args()


if __name__ == "__main__":
    run_frozen_training(parse_args())
