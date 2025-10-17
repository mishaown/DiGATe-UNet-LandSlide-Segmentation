import os
import numpy as np
from datetime import datetime
from .config import Config

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.smp_loss import SegmentationLosses
from models.smp_metrics import compute_dice_score, compute_iou_score

criterion = SegmentationLosses(loss_name='tversky', mode='binary', alpha=0.3, beta=0.7)

def run_epoch(loader, model: nn.Module, cfg: Config, optimizer=None, training: bool = False):
    model.train() if training else model.eval()
    prefix = "Train" if training else "Val"

    losses, dices, ious = [], [], []
    pbar = tqdm(loader, desc=prefix, leave=False)

    for x, y in pbar:
        x, y = prep_batch(x, y, cfg.device)

        with torch.set_grad_enabled(training):
            out = model(x)

            if isinstance(out, (tuple, list)):
                main = out[0]
                aux2 = out[1] if len(out) > 1 else None
                aux3 = out[2] if len(out) > 2 else None
            else:
                main, aux2, aux3 = out, None, None, None

            loss_main = criterion(main, y)
            loss_aux2 = criterion(aux2, y) if aux2 is not None else 0.0
            loss_aux3 = criterion(aux3, y) if aux3 is not None else 0.0


            loss = loss_main + 0.6 * loss_aux2  + 0.4 * loss_aux3

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            preds = (torch.sigmoid(main) > 0.5).float()

        losses.append(loss.item())
        dices.append(compute_dice_score(preds, y).item())
        ious.append(compute_iou_score(preds, y).item())

        pbar.set_postfix(
            loss=f"{losses[-1]:.4f}",
            dice=f"{dices[-1]:.4f}",
            iou=f"{ious[-1]:.4f}",
        )

    return np.mean(losses), np.mean(dices), np.mean(ious)

""""
Single Stream Training Strategy
"""

def train_single(train_dataset, val_dataset, model: nn.Module, cfg: Config):
    os.makedirs(os.path.dirname(cfg.model_save_path) or ".", exist_ok=True)
    device = torch.device(cfg.device)
    model.to(device)

    train_loader, val_loader = get_dataloaders(train_dataset, val_dataset, cfg)
    optimizer = get_optimizer(model, cfg)

    best_val_dice = 0.0
    history = {"train_loss":[], "train_dice":[], "train_iou":[], "val_loss":[], "val_dice":[], "val_iou":[]}

    start = datetime.now()
    print(f"Starting training at {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {cfg.device}")

    for epoch in range(1, cfg.num_epochs+1):
        tl, td, ti = run_epoch(train_loader, model, cfg, optimizer, training=True)
        vl, vd, vi = run_epoch(val_loader,   model, cfg, optimizer=None,  training=False)

        print(f"[{epoch}/{cfg.num_epochs}] " f"Train loss={tl:.4f}, dice={td:.4f}, iou={ti:.4f} | " f"Val   loss={vl:.4f}, dice={vd:.4f}, iou={vi:.4f}")

        history["train_loss"].append(tl)
        history["train_dice"].append(td)
        history["train_iou"].append(ti)
        history["val_loss"].append(vl)
        history["val_dice"].append(vd)
        history["val_iou"].append(vi)

        if vd > best_val_dice:
            best_val_dice = vd
            torch.save(model.state_dict(), cfg.model_save_path)
            print(f"  ↳ Saved new best model (dice={vd:.4f})")

    end = datetime.now()
    print(f"Finished at {end.strftime('%Y-%m-%d %H:%M:%S')} (duration {end-start})")
    return history

def get_dataloaders(train_ds, val_ds, cfg: Config):
    num_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return train_loader, val_loader

def prep_batch(x, y, device):
    x = x.float().to(device, non_blocking=True)
    if y.dtype.is_floating_point:
        y = y.round().long()
    y = y.to(device, non_blocking=True)
    if y.dim() == 3:
        y = y.unsqueeze(1)
    return x, y

def get_optimizer(model: nn.Module, cfg: Config):
    return Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)