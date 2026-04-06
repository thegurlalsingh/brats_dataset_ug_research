import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import nibabel as nib  # just in case

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        # pred: (B, 3, D, H, W) logits  → we only supervise the central slice (D=3 → index 1)
        central_pred = pred[:, :, 1]          # (B, 3, H, W)
        loss_bce = self.bce(central_pred, target)
        loss_dice = self.dice(central_pred, target)
        return loss_bce + 0.5 * loss_dice   # you can tune the 0.5 weight