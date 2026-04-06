import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os

from dataloader1       import BraTSDataset
from preprocess2       import Hybrid2_5DBackbone
from sparse_selection4 import SparseSelectionModule
from volumetric_context5 import DynamicSparseRefinerModel
from fusion_layer6     import FusionLayer

# =========================================================
# 🔥 CONFIG
# =========================================================
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS     = 10
LR         = 1e-4
VAL_INTERVAL = 1
ROOT_DIR   = r"C:\Users\Gurlal-Stu\Downloads\custom\data"
SAVE_DIR   = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================================================
# 🔹 LOSSES
#
# FIX: DiceLoss now operates correctly for both 2D (B,C,H,W)
# and 3D (B,C,D,H,W) tensors by flattening spatial dims.
# ComboLoss still uses BCEWithLogitsLoss (multi-label correct).
# =========================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # Flatten all spatial dims (works for both 2D and 3D preds)
        pred   = pred.flatten(2)    # (B, C, -1)
        target = target.flatten(2)  # (B, C, -1)

        intersection = (pred * target).sum(dim=2)
        union        = pred.sum(dim=2) + target.sum(dim=2)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # target must match pred shape for BCE
        return self.dice(pred, target) + self.bce(pred, target)


# =========================================================
# 🔹 METRICS
# =========================================================
def dice_score(pred, target, threshold=0.5):
    """Works for both 2D and 3D predictions."""
    pred = (torch.sigmoid(pred) > threshold).float()

    pred   = pred.flatten(2)
    target = target.flatten(2)

    intersection = (pred * target).sum(dim=2)
    union        = pred.sum(dim=2) + target.sum(dim=2)

    dice = (2. * intersection + 1e-5) / (union + 1e-5)
    return dice.mean().item()


# =========================================================
# 🔥 FULL MODEL
#
# KEY FIXES vs original:
#   1. uncertainty computed from proper sigmoid entropy
#      (not variance of sigmoid — that double-sigmoid was wrong).
#   2. refined_2d extracted from centre slice of 3D output,
#      then spatially aligned to backbone feature size.
#   3. Auxiliary coarse 3D loss: the 3D coarse pred is
#      supervised against a tiled 3D target to provide signal
#      to the 3D branch without shape mismatches.
#   4. 2D coarse head removed — we only have one 2D coarse
#      prediction (from the final fusion), which is cleaner.
#   5. importance head kept as sigmoid output.
# =========================================================
class FullModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone  = Hybrid2_5DBackbone(k=3, base_channels=64)
        self.sparse    = SparseSelectionModule(64, patch_size=8, num_slices=3)
        self.refiner3d = DynamicSparseRefinerModel(self.sparse, num_classes=3, iters=2)
        self.fusion    = FusionLayer(64, num_classes=3)

        # 2D heads on backbone features
        self.coarse_head = nn.Conv2d(64, 3, 1)
        self.imp_head    = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        """
        x : (B, 12, H, W)
        Returns:
            final_pred  : (B, 3, H, W)   — main output
            coarse_2d   : (B, 3, H, W)   — auxiliary 2D loss
            coarse_3d   : (B, 3, D, H, W)— auxiliary 3D loss
        """
        B, _, H, W = x.shape

        # ── 2D Backbone ────────────────────────────────────────
        features  = self.backbone(x)            # (B, 64, H, W)
        coarse_2d = self.coarse_head(features)  # (B, 3, H, W)
        importance = torch.sigmoid(self.imp_head(features))  # (B, 1, H, W)

        # ── 3D Branch ──────────────────────────────────────────
        refiner_out = self.refiner3d(x)
        final_3d    = refiner_out["final"]      # (B, 3, D, H, W)
        coarse_3d   = refiner_out["coarse"]     # (B, 3, D, Hf, Wf)
        feat3d      = refiner_out["feat3d"]     # (B, 64, D, Hf, Wf)

        # Extract centre slice features for fusion
        D = feat3d.shape[2]
        refined_2d = feat3d[:, :, D // 2]      # (B, 64, Hf, Wf)

        # Align refined_2d to backbone feature spatial size if needed
        if refined_2d.shape[-2:] != (H, W):
            refined_2d = nn.functional.interpolate(
                refined_2d, size=(H, W), mode='bilinear', align_corners=False
            )

        # ── FIX 1: Proper uncertainty from coarse_2d ──────────
        # sigmoid entropy per channel, max across classes
        prob = torch.sigmoid(coarse_2d)                      # (B, 3, H, W)
        eps  = 1e-6
        entropy = -(prob * torch.log(prob + eps) +
                    (1 - prob) * torch.log(1 - prob + eps))  # (B, 3, H, W)
        unc, _ = torch.max(entropy, dim=1, keepdim=True)     # (B, 1, H, W)
        unc = (unc / (torch.log(torch.tensor(2.0)) + eps)).clamp(0, 1)

        # ── Fusion ─────────────────────────────────────────────
        final_pred = self.fusion(
            features, coarse_2d, importance, unc, refined_2d
        )   # (B, 3, H, W)

        return final_pred, coarse_2d, coarse_3d, final_3d


# =========================================================
# 🔹 TRAIN + VALIDATION
# =========================================================
def train():
    # ── Dataset ────────────────────────────────────────────
    full_dataset = BraTSDataset(ROOT_DIR, k=3)

    val_size   = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # FIX: num_workers=0 so the volume cache actually works.
    # With workers>0 each subprocess gets its own dataset copy
    # and the cache never hits → every __getitem__ reads from disk.
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # ── Model ──────────────────────────────────────────────
    model     = FullModel().to(DEVICE)
    criterion = ComboLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

        for x, y in loop:
            # y: (B, 3, H, W)  — 2D target from dataloader
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()

            final_pred, coarse_2d, coarse_3d, final_3d = model(x)

            # ── Main loss (2D final prediction) ────────────────
            loss = criterion(final_pred, y)

            # ── Auxiliary 2D coarse loss ────────────────────────
            loss = loss + 0.4 * criterion(coarse_2d, y)

            # ── Auxiliary 3D coarse loss ────────────────────────
            # Expand 2D target to 3D: (B,3,H,W) → (B,3,D,Hf,Wf)
            # We tile the same 2D label across all depth slices and
            # resize to coarse_3d's spatial size.
            D_c, Hc, Wc = coarse_3d.shape[2], coarse_3d.shape[3], coarse_3d.shape[4]
            y_3d = y.unsqueeze(2).expand(-1, -1, D_c, -1, -1)   # (B,3,D,H,W)
            if y_3d.shape[-2:] != (Hc, Wc):
                y_3d = nn.functional.interpolate(
                    y_3d.reshape(x.shape[0] * D_c, 3, y.shape[-2], y.shape[-1]),
                    size=(Hc, Wc), mode='nearest'
                ).reshape(x.shape[0], 3, D_c, Hc, Wc)
            loss = loss + 0.2 * criterion(coarse_3d, y_3d)

            # ── Auxiliary 3D final loss ─────────────────────────
            D_f, Hf, Wf = final_3d.shape[2], final_3d.shape[3], final_3d.shape[4]
            y_3d_final = y.unsqueeze(2).expand(-1, -1, D_f, -1, -1)
            if y_3d_final.shape[-2:] != (Hf, Wf):
                y_3d_final = nn.functional.interpolate(
                    y_3d_final.reshape(x.shape[0] * D_f, 3, y.shape[-2], y.shape[-1]),
                    size=(Hf, Wf), mode='nearest'
                ).reshape(x.shape[0], 3, D_f, Hf, Wf)
            loss = loss + 0.2 * criterion(final_3d, y_3d_final)

            loss.backward()

            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)
        scheduler.step()

        print(f"\nEpoch {epoch+1} — Train Loss: {train_loss:.4f}")

        # ── Validation ─────────────────────────────────────────
        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            val_loss = val_dice = 0

            with torch.no_grad():
                for x, y in tqdm(val_loader, desc="Validation"):
                    x, y = x.to(DEVICE), y.to(DEVICE)

                    final_pred, _, _, _ = model(x)

                    val_loss += criterion(final_pred, y).item()
                    val_dice += dice_score(final_pred, y)

            val_loss /= len(val_loader)
            val_dice /= len(val_loader)

            history["val_loss"].append(val_loss)
            history["val_dice"].append(val_dice)

            print(f"  Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

            # Per-class dice for debugging ET/TC/WT
            model.eval()
            class_dice = torch.zeros(3, device=DEVICE)
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    pred, _, _, _ = model(x)
                    pred_bin = (torch.sigmoid(pred) > 0.5).float()
                    for c in range(3):
                        inter = (pred_bin[:, c] * y[:, c]).sum()
                        union = pred_bin[:, c].sum() + y[:, c].sum()
                        class_dice[c] += ((2 * inter + 1e-5) / (union + 1e-5)).item()
            class_dice /= len(val_loader)
            print(f"  WT Dice: {class_dice[0]:.4f} | TC Dice: {class_dice[1]:.4f} | ET Dice: {class_dice[2]:.4f}")

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
                print("  ✅ Best model saved!")

    # ── Save final ──────────────────────────────────────────
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))
    np.save(os.path.join(SAVE_DIR, "history.npy"), history)
    print("\n✅ Training Complete!")

    # ── Plot ────────────────────────────────────────────────
    epochs     = range(1, len(history["train_loss"]) + 1)
    val_epochs = range(VAL_INTERVAL,
                       VAL_INTERVAL * len(history["val_loss"]) + 1,
                       VAL_INTERVAL)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs,     history["train_loss"], label="Train Loss")
    plt.plot(val_epochs, history["val_loss"],   label="Val Loss")
    plt.plot(val_epochs, history["val_dice"],   label="Val Dice")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Training Metrics")
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "training_curve.png"))
    plt.close()


# =========================================================
# 🚀 RUN
# =========================================================
if __name__ == "__main__":
    train()