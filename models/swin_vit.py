import os
from copy import deepcopy
from ouhands_loader import OuhandsDS
import torch
from torch.utils.data import DataLoader
from torchvision.models import swin_v2_t
import torch.nn as nn

def main():
    # Use correct dataset path when running from models folder
    dataset_root = '../dataset'
    train_ds = OuhandsDS(root_dir=dataset_root, split='train')
    val_ds = OuhandsDS(root_dir=dataset_root, split='validation') 
    test_ds = OuhandsDS(root_dir=dataset_root, split='test')

    batch_size = 32
    num_workers = 4

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("Using device:", device)

    num_classes = 10
    lr = 1e-4
    epochs = 500

    model = swin_v2_t(weights=None)  # Use weights=None instead of pretrained=False
    in_feats = model.head.in_features
    model.head = nn.Linear(in_feats, num_classes) 
    model.to(device)
    
    print(f"Model created: Swin Transformer V2 Tiny")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Input features: {in_feats}, Output classes: {num_classes}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_dir = "./checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_val_acc = -1.0
    best_epoch = -1

    def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val_acc):
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "num_classes": num_classes,
            "lr": lr,
        }, path)

    def train_one_epoch():
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        return total_loss / total, total_correct / total

    @torch.no_grad()
    def evaluate():
        model.eval()
        total_loss, total_correct, total = 0.0, 0, 0
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        return total_loss / total, total_correct / total

    print(f"\nStarting training for {epochs} epochs...")
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch()
        val_loss, val_acc = evaluate()
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:03d}/{epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"lr {current_lr:.2e}")

        # Save every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch:03d}.pt")
            save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, best_val_acc)
            print(f"  → Checkpoint saved: {ckpt_path}")

        # Save best by val acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_path = os.path.join(ckpt_dir, "best.pt")
            save_checkpoint(best_path, epoch, model, optimizer, scheduler, best_val_acc)
            print(f"  → New best model saved: val_acc {val_acc:.4f}")

    print(f"Best val acc: {best_val_acc:.4f} at epoch {best_epoch}")

if __name__ == "__main__":
    main()