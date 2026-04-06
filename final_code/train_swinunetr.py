"""
train_swinunetr.py — Swin UNETR training on BraTS 2023 GLI.

Differences vs SegResNet:
  • Uses MONAI SwinUNETR with gradient checkpointing (saves VRAM on A6000)
  • use_checkpoint=True in SWIN_CFG halves VRAM at ~15% compute cost
  • Identical loss / scheduler / early-stop / AMP logic as SegResNet
  • Checkpoint saved as swinunetr_best.pth (separate from SegResNet)
  • Optional pretrained SSL weights from MONAI model zoo

Run:
    python train_swinunetr.py
    python train_swinunetr.py --epochs 100 --lr 1e-4 --resume
    python train_swinunetr.py --pretrained    # load MONAI self-supervised weights
"""

import argparse
import csv
import logging
import os
import random
import time

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.utils import set_determinism

from config import (
    CHECKPOINT_DIR, RESULTS_DIR,
    SWIN_CFG, SPATIAL_SIZE,
    SW_ROI_SIZE, SW_OVERLAP,
    EPOCHS, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    LR_INITIAL, LR_MIN, WEIGHT_DECAY,
    LOSS_LAMBDA_DICE, LOSS_LAMBDA_FOCAL, FOCAL_GAMMA,
    USE_AMP, EARLY_STOP_PATIENCE, VAL_INTERVAL,
    MODALITY_KEYS, NUM_MODALITIES, NUM_CLASSES, REGION_NAMES,
    DEVICE, GLOBAL_SEED,
)
from dataset import get_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


# ──────────────────────────────────────────────────────────────────────────────
# SANITY ASSERTIONS
# ──────────────────────────────────────────────────────────────────────────────

def assert_config() -> None:
    assert NUM_MODALITIES == 4, \
        f"NUM_MODALITIES must be 4 for BraTS, got {NUM_MODALITIES}"
    assert NUM_CLASSES == 3, \
        f"NUM_CLASSES must be 3 (WT/TC/ET), got {NUM_CLASSES}"
    assert MODALITY_KEYS == ["t1n", "t1c", "t2w", "t2f"], \
        f"Modality order wrong: {MODALITY_KEYS}"
    assert SWIN_CFG["in_channels"] == NUM_MODALITIES, \
        "SWIN_CFG in_channels does not match NUM_MODALITIES"
    assert SWIN_CFG["out_channels"] == NUM_CLASSES, \
        "SWIN_CFG out_channels does not match NUM_CLASSES"
    # assert SWIN_CFG["img_size"] == SPATIAL_SIZE, \
    #     f"SWIN_CFG img_size {SWIN_CFG['img_size']} != SPATIAL_SIZE {SPATIAL_SIZE}"
    logger.info("Config assertions passed.")


# ──────────────────────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────────────────────

def build_model(pretrained: bool = False) -> nn.Module:
    """
    Build SwinUNETR.
    use_checkpoint=True in SWIN_CFG activates gradient checkpointing — 
    trades compute for memory, essential for fitting 128^3 volumes on 48GB.
    """
    model = SwinUNETR(**SWIN_CFG).to(DEVICE)

    if pretrained:
        # MONAI self-supervised pre-trained weights for SwinUNETR (BraTS domain)
        # Loads encoder weights only; decoder is randomly initialized and trained.
        try:
            from monai.networks.nets import SwinUNETR as _S
            weight_url = (
                "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/"
                "download/0.8.1/model_swinvit.pt"
            )
            import urllib.request, tempfile, os
            tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
            logger.info(f"Downloading pretrained SwinViT weights …")
            urllib.request.urlretrieve(weight_url, tmp.name)
            weights = torch.load(tmp.name, map_location=DEVICE)
            # Only load keys that match (encoder only)
            model_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in weights.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            os.unlink(tmp.name)
            logger.info(
                f"Loaded {len(pretrained_dict)}/{len(model_dict)} "
                f"layers from pretrained SwinViT."
            )
        except Exception as e:
            logger.warning(f"Pretrained weight loading failed: {e}. Training from scratch.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"SwinUNETR built — {n_params / 1e6:.2f}M trainable parameters")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# LOSS — identical rationale to SegResNet
# ──────────────────────────────────────────────────────────────────────────────

def build_loss() -> nn.Module:
    """
    DiceFocalLoss with sigmoid.
    BraTS regions are multi-label (ET ⊆ TC ⊆ WT) — sigmoid is correct.
    Softmax would incorrectly force competition between region channels.
    """
    return DiceFocalLoss(
        sigmoid=True,
        lambda_dice=LOSS_LAMBDA_DICE,
        lambda_focal=LOSS_LAMBDA_FOCAL,
        gamma=FOCAL_GAMMA,
    )


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN ONE EPOCH
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:     nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn:   nn.Module,
    scaler:    GradScaler,
    epoch:     int,
    epoch_bar: tqdm,
) -> float:
    model.train()
    epoch_loss = 0.0
    n_steps    = 0

    batch_bar = tqdm(
        loader,
        desc="  Train batches",
        leave=False,
        unit="batch",
        dynamic_ncols=True,
        colour="green",
    )

    for batch_idx, batch in enumerate(batch_bar):
        images = batch["image"].to(DEVICE, non_blocking=True)   # (B,4,128,128,128)
        labels = batch["label"].to(DEVICE, non_blocking=True)   # (B,3,128,128,128)

        if epoch == 1 and batch_idx == 0:
            assert images.shape[1] == NUM_MODALITIES, \
                f"Image channels: expected {NUM_MODALITIES}, got {images.shape[1]}"
            assert labels.shape[1] == NUM_CLASSES, \
                f"Label channels: expected {NUM_CLASSES}, got {labels.shape[1]}"
            logger.info(
                f"First batch — image: {tuple(images.shape)}  "
                f"label: {tuple(labels.shape)}"
            )

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=USE_AMP):
            logits = model(images)          # (B,3,128,128,128)
            loss   = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        n_steps    += 1
        running_avg = epoch_loss / n_steps

        batch_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_avg:.4f}")
        epoch_bar.set_postfix(batch_loss=f"{running_avg:.4f}")

    batch_bar.close()
    return epoch_loss / max(n_steps, 1)


# ──────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def validate(model: nn.Module, loader, dice_metric: DiceMetric) -> tuple:
    model.eval()
    dice_metric.reset()

    post_sigmoid = nn.Sigmoid()
    post_thresh  = AsDiscrete(threshold=0.5)

    val_bar = tqdm(
        loader,
        desc="  Validating",
        leave=False,
        unit="vol",
        dynamic_ncols=True,
        colour="cyan",
    )

    with torch.no_grad():
        for batch in val_bar:
            images = batch["image"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            with autocast("cuda", enabled=USE_AMP):
                logits = sliding_window_inference(
                    inputs        = images,
                    roi_size      = SW_ROI_SIZE,
                    sw_batch_size = 1,
                    predictor     = model,
                    overlap       = SW_OVERLAP,
                    mode          = "gaussian",
                )

            probs = post_sigmoid(logits)
            preds = torch.stack(
                [post_thresh(probs[0, c]) for c in range(NUM_CLASSES)],
                dim=0,
            ).unsqueeze(0)

            dice_metric(y_pred=preds, y=labels)

            current = dice_metric.aggregate().tolist()
            val_bar.set_postfix(
                WT=f"{current[0]:.3f}",
                TC=f"{current[1]:.3f}",
                ET=f"{current[2]:.3f}",
            )

    val_bar.close()

    scores    = dice_metric.aggregate().tolist()
    mean_dice = float(np.mean(scores))
    dice_metric.reset()
    return mean_dice, scores


# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module, optimizer, epoch: int, best_dice: float
) -> None:
    path = CHECKPOINT_DIR / "swinunetr_best.pth"
    torch.save(
        {
            "epoch":           epoch,
            "best_dice":       best_dice,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {
                "in_channels":   NUM_MODALITIES,
                "out_channels":  NUM_CLASSES,
                "modality_keys": MODALITY_KEYS,
                "img_size":      SPATIAL_SIZE,
            },
        },
        path,
    )
    logger.info(f"  ✓ Best checkpoint saved → {path}  (epoch {epoch} | dice {best_dice:.4f})")


def load_checkpoint(
    model: nn.Module, optimizer=None
) -> tuple:
    path = CHECKPOINT_DIR / "swinunetr_best.pth"
    if not path.exists():
        logger.info("No checkpoint found — starting from scratch.")
        return model, optimizer, 0, 0.0
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    logger.info(
        f"Resumed from {path} "
        f"(epoch {ckpt['epoch']} | best dice {ckpt['best_dice']:.4f})"
    )
    return model, optimizer, ckpt["epoch"], ckpt["best_dice"]


# ──────────────────────────────────────────────────────────────────────────────
# CURVES
# ──────────────────────────────────────────────────────────────────────────────

def save_curves(history: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path = RESULTS_DIR / "swinunetr_training_curve.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    logger.info(f"Curve CSV  → {csv_path}")

    val_rows  = [h for h in history if h.get("val_mean_dice") is not None]
    ep_all    = [h["epoch"]        for h in history]
    loss_all  = [h["train_loss"]   for h in history]
    ep_val    = [h["epoch"]        for h in val_rows]
    dice_mean = [h["val_mean_dice"] for h in val_rows]
    dice_wt   = [h["val_dice_WT"]  for h in val_rows]
    dice_tc   = [h["val_dice_TC"]  for h in val_rows]
    dice_et   = [h["val_dice_ET"]  for h in val_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SwinUNETR — Training Curves", fontsize=14, fontweight="bold")

    ax1.plot(ep_all, loss_all, color="#E24B4A", linewidth=1.8, label="Train Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("DiceFocal Loss")
    ax1.set_title("Training Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ep_val, dice_mean, color="#1D9E75", linewidth=2.2, label="Mean Dice")
    ax2.plot(ep_val, dice_wt,   color="#378ADD", linewidth=1.4, linestyle="--", label="WT")
    ax2.plot(ep_val, dice_tc,   color="#EF9F27", linewidth=1.4, linestyle="--", label="TC")
    ax2.plot(ep_val, dice_et,   color="#D4537E", linewidth=1.4, linestyle="--", label="ET")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Dice Score")
    ax2.set_title("Validation Dice — WT / TC / ET")
    ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    png_path = RESULTS_DIR / "swinunetr_training_curve.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Curve PNG  → {png_path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main(args) -> None:
    seed_everything()
    assert_config()

    logger.info("=" * 65)
    logger.info("  SwinUNETR — BraTS 2023 GLI  |  Training")
    logger.info(f"  Device      : {DEVICE}")
    logger.info(f"  Epochs      : {args.epochs}")
    logger.info(f"  Batch size  : {args.batch_size}")
    logger.info(f"  LR          : {args.lr}")
    logger.info(f"  AMP         : {USE_AMP}")
    logger.info(f"  Grad ckpt   : {SWIN_CFG.get('use_checkpoint', False)}")
    logger.info(f"  Loss        : DiceFocal (λ_dice={LOSS_LAMBDA_DICE}, λ_focal={LOSS_LAMBDA_FOCAL})")
    logger.info("=" * 65)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    safe_workers = 0 if os.name == "nt" else NUM_WORKERS
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size  = args.batch_size,
        num_workers = safe_workers,
        pin_memory  = PIN_MEMORY,
    )
    logger.info(
        f"Loaders ready — "
        f"train: {len(train_loader.dataset)} | "
        f"val: {len(val_loader.dataset)}"
    )

    # ── Model / loss / optimizer ──────────────────────────────────────────────
    model     = build_model(pretrained=args.pretrained)
    loss_fn   = build_loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=LR_MIN
    )
    scaler = GradScaler("cuda", enabled=USE_AMP)

    dice_metric = DiceMetric(
        include_background=True,
        reduction="mean_batch",
        get_not_nans=False,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_dice   = 0.0
    if args.resume:
        model, optimizer, start_epoch, best_dice = load_checkpoint(model, optimizer)

    # ── Loop ──────────────────────────────────────────────────────────────────
    no_improve = 0
    history    = []

    epoch_bar = tqdm(
        range(start_epoch + 1, args.epochs + 1),
        desc="Epochs",
        unit="epoch",
        dynamic_ncols=True,
        colour="blue",
    )

    for epoch in epoch_bar:
        t0 = time.time()
        epoch_bar.set_description(f"Epoch {epoch:03d}/{args.epochs}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, epoch, epoch_bar
        )
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        row = {
            "epoch":         epoch,
            "train_loss":    round(train_loss, 6),
            "lr":            round(lr_now, 8),
            "val_mean_dice": None,
            "val_dice_WT":   None,
            "val_dice_TC":   None,
            "val_dice_ET":   None,
        }

        if epoch % VAL_INTERVAL == 0 or epoch == args.epochs:
            mean_dice, per_region = validate(model, val_loader, dice_metric)

            row.update({
                "val_mean_dice": round(mean_dice, 4),
                "val_dice_WT":   round(per_region[0], 4),
                "val_dice_TC":   round(per_region[1], 4),
                "val_dice_ET":   round(per_region[2], 4),
            })

            epoch_bar.set_postfix(
                loss=f"{train_loss:.4f}",
                dice=f"{mean_dice:.4f}",
                WT=f"{per_region[0]:.3f}",
                TC=f"{per_region[1]:.3f}",
                ET=f"{per_region[2]:.3f}",
                lr=f"{lr_now:.1e}",
                best=f"{best_dice:.4f}",
            )

            logger.info(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Loss {train_loss:.4f} | "
                f"Dice {mean_dice:.4f} "
                f"[WT {per_region[0]:.3f}  TC {per_region[1]:.3f}  ET {per_region[2]:.3f}] | "
                f"LR {lr_now:.2e} | {time.time()-t0:.1f}s"
            )

            if mean_dice > best_dice:
                best_dice = mean_dice
                save_checkpoint(model, optimizer, epoch, best_dice)
                no_improve = 0
                epoch_bar.write(f"  ✓ New best Dice = {best_dice:.4f}  (epoch {epoch})")
            else:
                no_improve += 1
                logger.info(
                    f"  No improvement {no_improve}/{EARLY_STOP_PATIENCE} "
                    f"(best={best_dice:.4f})"
                )

            if no_improve >= EARLY_STOP_PATIENCE:
                epoch_bar.write(
                    f"  Early stopping at epoch {epoch} — "
                    f"no improvement for {EARLY_STOP_PATIENCE} val checks."
                )
                history.append(row)
                break
        else:
            epoch_bar.set_postfix(
                loss=f"{train_loss:.4f}",
                lr=f"{lr_now:.1e}",
            )
            logger.info(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Loss {train_loss:.4f} | "
                f"LR {lr_now:.2e} | {time.time()-t0:.1f}s"
            )

        history.append(row)

    epoch_bar.close()

    save_curves(history)
    logger.info("=" * 65)
    logger.info(f"  Done. Best val Dice = {best_dice:.4f}")
    logger.info(f"  Checkpoint → {CHECKPOINT_DIR / 'swinunetr_best.pth'}")
    logger.info("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SwinUNETR on BraTS 2023 GLI")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS,     help="Total training epochs")
    parser.add_argument("--batch_size", type=int,   default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr",         type=float, default=LR_INITIAL, help="Initial learning rate")
    parser.add_argument("--resume",     action="store_true",            help="Resume from best checkpoint")
    parser.add_argument("--pretrained", action="store_true",            help="Load MONAI SSL pretrained weights")
    args = parser.parse_args()
    main(args)