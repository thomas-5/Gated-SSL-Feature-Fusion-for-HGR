import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from landmarks_loader import OuhandsLandmarks
from mobilenet_v2 import MobileNetV2Landmark
from training_utils import train_one_epoch, evaluate

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_PATH = './'

def main():
    print(f'Using device: {DEVICE}')
    
    # Training hyperparameters
    num_workers = 4 if DEVICE == 'cuda' else 2
    learning_rate = 3e-3
    epochs = 50
    batch_size = 64 if DEVICE == 'cuda' else 16
    weight_decay = 1e-4

    # Create datasets
    print("Creating datasets...")
    train_ds = OuhandsLandmarks(
        split='train',
        augment=True,
        standardize=True
    )
    val_ds = OuhandsLandmarks(
        split='validation', 
        augment=False,
        standardize=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE == 'cuda'),
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE == 'cuda')
    )
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        
    # Create model
    model = MobileNetV2Landmark(pretrained=False).to(DEVICE)
    
    # Optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )   
    loss_fn = nn.SmoothL1Loss(beta=0.02) 
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler(device="cuda", enabled=(DEVICE == "cuda"))

    # Training loop
    best_val = float('inf')
    best_path = None
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # Train
        tr_loss, tr_mpjpe = train_one_epoch(
            model, train_loader, optimizer, scaler, DEVICE, loss_fn
        )
        
        # Validate
        val_loss, val_mpjpe = evaluate(model, val_loader, DEVICE, loss_fn)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch:03d} | "
              f"train_loss {tr_loss:.5f} mpjpe {tr_mpjpe:.5f} | "
              f"val_loss {val_loss:.5f} mpjpe {val_mpjpe:.5f} | "
              f"lr {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        score = val_mpjpe
        if score < best_val:
            best_val = score
            best_path = os.path.join(OUT_PATH, "mnv2_landmarks_best.pth")
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_mpjpe": val_mpjpe,
                "train_loss": tr_loss,
                "train_mpjpe": tr_mpjpe,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, best_path)
            print(f"[ckpt] saved {best_path} (val_mpjpe={val_mpjpe:.5f})")
        
        # Save latest model every 10 epochs
        if epoch % 10 == 0:
            latest_path = os.path.join(OUT_PATH, f"mnv2_landmarks_epoch_{epoch:03d}.pth")
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_mpjpe": val_mpjpe,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, latest_path)
            print(f"[ckpt] saved checkpoint {latest_path}")
    
    print(f"[done] best val_mpjpe: {best_val:.5f}, saved at: {best_path}")

if __name__ == '__main__':
    main()