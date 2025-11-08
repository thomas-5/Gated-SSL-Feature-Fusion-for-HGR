import torch
import torch.nn as nn

def mpjpe_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean per-joint position error (L1) over 21 joints, on normalized coords.
    pred/target: [B,63]
    """
    B = pred.shape[0]
    pred = pred.view(B, 21, 3)
    target = target.view(B, 21, 3)
    return (pred - target).abs().mean()

def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn):
    model.train()
    total_loss = 0.0
    total_mpjpe = 0.0
    n = 0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            preds = model(imgs)
            loss = loss_fn(preds, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            mp = mpjpe_l1(preds, targets)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_mpjpe += mp.item() * bs
        n += bs

    return total_loss / n, total_mpjpe / n

@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_mpjpe = 0.0
    n = 0
    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(imgs)
        loss = loss_fn(preds, targets)
        mp = mpjpe_l1(preds, targets)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_mpjpe += mp.item() * bs
        n += bs
    return total_loss / n, total_mpjpe / n

