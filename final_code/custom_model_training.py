"""
train_custom.py — training script for the custom BraTS model.

Uses the same train/val/test split as SegResNet and SwinUNETR (from dataset.py).
The custom model uses 2.5D slice-based inputs via BraTS25DDataset.
Losses: DiceFocalLoss on final output + auxiliary losses on coarse and vol_pred.
All checkpoints and curves saved to config.CHECKPOINT_DIR.
"""

import json
import logging
import random
import time
from pathlib import Path

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_DIR, RESULTS_DIR, SPLIT_CACHE,
    EPOCHS, LR_INITIAL, LR_MIN, WEIGHT_DECAY,
    VAL_INTERVAL, EARLY_STOP_PATIENCE,
    USE_AMP, DEVICE, GLOBAL_SEED, MODALITY_KEYS,
    NUM_CLASSES, REGION_NAMES, DATA_ROOT,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
)
from dataset import discover_cases, build_splits, get_case_paths_for_split
from custom_model_sanity_check import BraTS25DDataset, CustomBraTSModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# Must be at module level — Windows spawn-based multiprocessing cannot pickle
# local/nested functions, so worker_init_fn must live at the top of the file.
def _seed_worker(worker_id: int) -> None:
    np.random.seed(GLOBAL_SEED + worker_id)
    random.seed(GLOBAL_SEED + worker_id)

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)

MODEL_NAME  = "custom"
CKPT_BEST   = CHECKPOINT_DIR / f"{MODEL_NAME}_best.pth"
CKPT_LAST   = CHECKPOINT_DIR / f"{MODEL_NAME}_last.pth"
HISTORY_OUT = RESULTS_DIR    / f"{MODEL_NAME}_history.json"

# Custom-model specific settings
K           = 3          # number of adjacent slices
TARGET_HW   = (192, 192) # 2.5D spatial size
CUSTOM_LR   = 1e-4
CUSTOM_BS   = 4          # slices are cheap vs full 3D volumes
AUX_WEIGHT  = 0.4        # weight for auxiliary losses


# ===========================================================================
# LOSS
# ===========================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num   = 2 * (probs * targets).sum(dim=(2, 3)) + self.smooth
        den   = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth
        return 1 - (num / den).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.bce   = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = self.bce(logits, targets)
        prob = torch.sigmoid(logits)
        pt   = targets * prob + (1 - targets) * (1 - prob)
        return ((1 - pt) ** self.gamma * bce).mean()


class DiceFocalLoss(nn.Module):
    def __init__(self, lam_dice: float = 1.0, lam_focal: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.dice  = DiceLoss()
        self.focal = FocalLoss(gamma)
        self.ld, self.lf = lam_dice, lam_focal

    def forward(self, logits, targets):
        return self.ld * self.dice(logits, targets) + self.lf * self.focal(logits, targets)


# ===========================================================================
# DICE METRIC
# ===========================================================================

def dice_score(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    probs = (torch.sigmoid(logits) > 0.5).float()
    num = 2 * (probs * targets).sum(dim=(2, 3)) + smooth
    den = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth
    return (num / den).mean(dim=0)   # (3,) per region


# ===========================================================================
# TRAINING LOOP
# ===========================================================================

def train_one_epoch(model, loader, optimizer, loss_fn, scaler, epoch, total_epochs):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{total_epochs} [train]", leave=False, unit="batch")
    for batch in pbar:
        imgs   = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=USE_AMP):
            out = model(imgs)
            loss_main   = loss_fn(out["final"],    labels)
            loss_coarse = loss_fn(out["coarse"],   labels)

            vol_pred = out["vol_pred"]
            if vol_pred.shape[-2:] != labels.shape[-2:]:
                vol_pred = torch.nn.functional.interpolate(
                    vol_pred, labels.shape[-2:], mode='bilinear', align_corners=False)
            loss_vol  = loss_fn(vol_pred, labels)

            loss = loss_main + AUX_WEIGHT * (loss_coarse + loss_vol)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, loss_fn, epoch, total_epochs):
    model.eval()
    total_loss = 0.0
    dice_accum = torch.zeros(NUM_CLASSES, device=DEVICE)
    n = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{total_epochs} [val]  ", leave=False, unit="batch")
    for batch in pbar:
        imgs   = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)

        with autocast("cuda", enabled=USE_AMP):
            out  = model(imgs)
            loss = loss_fn(out["final"], labels)

        total_loss += loss.item()
        dice_accum += dice_score(out["final"], labels)
        n += 1
        pbar.set_postfix(val_loss=f"{loss.item():.4f}")

    mean_dice = dice_accum / max(n, 1)
    return total_loss / max(n, 1), mean_dice


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Training custom model — {EPOCHS} epochs")

    # ── Data ─────────────────────────────────────────────────────────────────
    all_cases = discover_cases(DATA_ROOT)
    splits    = build_splits(all_cases)

    train_dirs = get_case_paths_for_split(splits["train"], DATA_ROOT)
    val_dirs   = get_case_paths_for_split(splits["val"],   DATA_ROOT)

    logger.info(f"Train cases: {len(train_dirs)} | Val cases: {len(val_dirs)}")

    train_ds = BraTS25DDataset(train_dirs, k=K, target_size=TARGET_HW, augment=True)
    val_ds   = BraTS25DDataset(val_dirs,   k=K, target_size=TARGET_HW, augment=False)

    g = torch.Generator(); g.manual_seed(GLOBAL_SEED)

    train_loader = DataLoader(
        train_ds, batch_size=CUSTOM_BS, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        worker_init_fn=_seed_worker, generator=g,
        persistent_workers=(NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=CUSTOM_BS, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CustomBraTSModel(k=K, base_channels=64, patch_size=8).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Custom model parameters: {total_params:,}")

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=CUSTOM_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)
    scaler    = GradScaler("cuda", enabled=USE_AMP)
    loss_fn   = DiceFocalLoss()

    # ── Training ──────────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_dice_wt": [], "val_dice_tc": [], "val_dice_et": []}
    best_dice = 0.0
    no_improve = 0

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, epoch, EPOCHS)
        scheduler.step()

        history["train_loss"].append(train_loss)

        if epoch % VAL_INTERVAL == 0 or epoch == EPOCHS:
            val_loss, val_dice = validate(model, val_loader, loss_fn, epoch, EPOCHS)
            mean_val_dice = val_dice.mean().item()

            history["val_loss"].append(val_loss)
            history["val_dice_wt"].append(val_dice[0].item())
            history["val_dice_tc"].append(val_dice[1].item())
            history["val_dice_et"].append(val_dice[2].item())

            epoch_bar.set_postfix({
                "tr_loss": f"{train_loss:.4f}",
                "vl_loss": f"{val_loss:.4f}",
                "WT":      f"{val_dice[0]:.3f}",
                "TC":      f"{val_dice[1]:.3f}",
                "ET":      f"{val_dice[2]:.3f}",
                "best":    f"{best_dice:.3f}",
            })

            logger.info(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"Dice WT={val_dice[0]:.3f} TC={val_dice[1]:.3f} ET={val_dice[2]:.3f} | "
                f"mean={mean_val_dice:.3f} | {time.time()-t0:.1f}s"
            )

            if mean_val_dice > best_dice:
                best_dice = mean_val_dice
                no_improve = 0
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "best_dice": best_dice}, CKPT_BEST)
                logger.info(f"  *** New best saved: {best_dice:.4f}")
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break
        else:
            epoch_bar.set_postfix({"tr_loss": f"{train_loss:.4f}"})
            logger.info(f"Epoch {epoch:3d}/{EPOCHS} | train={train_loss:.4f} | {time.time()-t0:.1f}s")

    # Save last checkpoint and history
    torch.save({"epoch": epoch, "model": model.state_dict()}, CKPT_LAST)
    with open(HISTORY_OUT, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete. Best val mean Dice: {best_dice:.4f}")
    logger.info(f"Checkpoint: {CKPT_BEST}")
    logger.info(f"History:    {HISTORY_OUT}")

    # Save training curve plot
    _save_training_curve(history)


def _save_training_curve(history: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(history["train_loss"], label="train loss")
        if history["val_loss"]:
            val_x = list(range(VAL_INTERVAL - 1, len(history["train_loss"]), VAL_INTERVAL))
            val_x = val_x[:len(history["val_loss"])]
            axes[0].plot(val_x, history["val_loss"], label="val loss")
        axes[0].set_title("Custom model — loss")
        axes[0].set_xlabel("Epoch"); axes[0].legend()

        for region, key in zip(REGION_NAMES, ["val_dice_wt", "val_dice_tc", "val_dice_et"]):
            axes[1].plot(history[key], label=f"Dice {region}")
        axes[1].set_title("Custom model — val Dice")
        axes[1].set_xlabel("Val step"); axes[1].set_ylim(0, 1); axes[1].legend()

        plt.tight_layout()
        out = RESULTS_DIR / "custom_training_curve.png"
        plt.savefig(out, dpi=150)
        plt.close()
        logger.info(f"Training curve saved: {out}")
    except Exception as e:
        logger.warning(f"Could not save training curve: {e}")


if __name__ == "__main__":
    main()