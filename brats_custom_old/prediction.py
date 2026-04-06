import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# ── Import your modules ───────────────────────────────
from dataloader1 import BraTSDataset
from preprocess2 import Hybrid2_5DBackbone
from sparse_selection4 import SparseSelectionModule
from volumetric_context5 import DynamicSparseRefinerModel
from fusion_layer6 import FusionLayer
from main import FullModel, dice_score  # assuming your FullModel and dice_score are in train.py

# =========================================================
# 🔥 CONFIG
# =========================================================
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
ROOT_DIR   = r"C:\Users\Gurlal-Stu\Downloads\brats_nnuNet\data"
CHECKPOINT = "checkpoints/best_model.pth"  # path to your checkpoint

# =========================================================
# 🔹 HELPER: Safe state dict load
# =========================================================
def load_checkpoint_safe(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()
    filtered_dict = {}

    for k, v in checkpoint.items():
        if k in model_dict:
            if v.size() == model_dict[k].size():
                filtered_dict[k] = v
            else:
                print(f"Skipping {k}: shape mismatch {v.size()} vs {model_dict[k].size()}")
        else:
            print(f"Skipping {k}: not in model")

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print("✅ Checkpoint loaded safely (partial load if needed)")

# =========================================================
# 🔹 TEST FUNCTION
# =========================================================
def test():
    # ── Dataset
    test_ds = BraTSDataset(ROOT_DIR, k=3)  # or use a separate test folder
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ── Model
    model = FullModel().to(DEVICE)
    load_checkpoint_safe(model, CHECKPOINT, DEVICE)
    model.eval()

    dice_scores = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            final_pred, _, _, _ = model(x)

            # Compute dice score for each sample
            dice = dice_score(final_pred, y)
            dice_scores.append(dice)

    dice_scores = np.array(dice_scores)
    print(f"\nAverage Dice Score: {dice_scores.mean():.4f}")
    print(f"Min Dice Score: {dice_scores.min():.4f} | Max Dice Score: {dice_scores.max():.4f}")

    # Save predictions if needed
    # Example: save first 3 predictions
    for i, (x, y) in enumerate(test_loader):
        if i >= 3:
            break
        x = x.to(DEVICE)
        pred, _, _, _ = model(x)
        pred_np = torch.sigmoid(pred).cpu().numpy()  # (B,3,H,W)
        np.save(f"prediction_{i}.npy", pred_np)
        print(f"Saved prediction_{i}.npy")

if __name__ == "__main__":
    test()