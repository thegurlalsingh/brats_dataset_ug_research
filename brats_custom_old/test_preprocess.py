import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


# ===================================================================
# 1. Coarse Head
# ===================================================================
class CoarseHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        return self.net(x)


# ===================================================================
# 2. Importance Head
# ===================================================================
class ImportanceHead(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


# ===================================================================
# 3. Uncertainty (Improved)
# ===================================================================
def compute_uncertainty(logits):
    """Entropy-based uncertainty for multi-class segmentation"""
    probs = torch.softmax(logits, dim=1)
    uncertainty = -probs * torch.log(probs + 1e-8)
    uncertainty = uncertainty.sum(dim=1, keepdim=True)   # (B, 1, H, W)
    return uncertainty


# ===================================================================
# 4. Structured Sparse Selector (Core of your idea)
# ===================================================================
class SparseSelector(nn.Module):
    """
    Takes importance and uncertainty → returns structured patches + coordinates
    Avoids scattered pixels using connected components + patch extraction
    """
    def __init__(self, top_k=8, patch_size=64, min_region_size=25):
        super().__init__()
        self.top_k = top_k
        self.patch_size = patch_size
        self.min_region_size = min_region_size

    def forward(self, importance, uncertainty, coarse_logits=None):
        """
        importance: (B, 1, H, W)
        uncertainty: (B, 1, H, W)
        coarse_logits: (B, 3, H, W) optional
        """
        B, _, H, W = importance.shape
        device = importance.device

        # Focus map = Importance * Uncertainty
        focus = importance * uncertainty

        # Optional: penalize over-confident areas
        if coarse_logits is not None:
            confidence = torch.softmax(coarse_logits, dim=1).max(dim=1, keepdim=True)[0]
            focus = focus * (1.0 - confidence.clamp(max=0.9))

        focus = focus.squeeze(1)  # (B, H, W)

        selected_patches = []
        patch_coords = []   # (batch_idx, x1, y1, x2, y2)

        for b in range(B):
            score_map = focus[b].cpu().numpy()

            # Binary + Connected Components
            binary = score_map > (score_map.mean() + 0.5 * score_map.std())
            labeled, num_labels = ndimage.label(binary)

            if num_labels == 0:
                # Fallback: center patch
                x1 = max(0, (W - self.patch_size) // 2)
                y1 = max(0, (H - self.patch_size) // 2)
                patch_coords.append((b, x1, y1, x1 + self.patch_size, y1 + self.patch_size))
                selected_patches.append(torch.zeros(1, 64, self.patch_size, self.patch_size, device=device))
                continue

            # Score each region
            regions = []
            for label in range(1, num_labels + 1):
                mask = (labeled == label)
                area = np.sum(mask)
                if area < self.min_region_size:
                    continue
                score = score_map[mask].mean()
                ys, xs = np.where(mask)
                cx, cy = int(xs.mean()), int(ys.mean())
                regions.append((score, cx, cy, area))

            # Take Top-K highest scoring regions
            regions.sort(reverse=True, key=lambda t: t[0])
            for score, cx, cy, _ in regions[:self.top_k]:
                half = self.patch_size // 2
                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(W, x1 + self.patch_size)
                y2 = min(H, y1 + self.patch_size)

                patch_coords.append((b, x1, y1, x2, y2))

                # Extract patch from features (will be passed later)
                # For now we store coordinates. Actual patch extraction will be done outside.
                dummy_patch = torch.zeros(1, 64, self.patch_size, self.patch_size, device=device)
                selected_patches.append(dummy_patch)

        return {
            "focus": focus,
            "selected_patches": selected_patches,   # list of dummy patches
            "patch_coords": patch_coords,           # list of (b, x1,y1,x2,y2)
            "importance": importance,
            "uncertainty": uncertainty
        }


# ===================================================================
# 5. Full Refinement Module (connects with your backbone)
# ===================================================================
class RefinementModule(nn.Module):
    def __init__(self, in_channels=64, num_classes=3, top_k=8, patch_size=64):
        super().__init__()
        self.coarse_head = CoarseHead(in_channels, num_classes)
        self.importance_head = ImportanceHead(in_channels)
        self.sparse_selector = SparseSelector(top_k=top_k, patch_size=patch_size)

    def forward(self, features):
        """
        features: (B, 64, H, W) from your 2.5D backbone
        """
        # Coarse prediction
        coarse_logits = self.coarse_head(features)

        # Importance & Uncertainty
        importance = self.importance_head(features)
        uncertainty = compute_uncertainty(coarse_logits)

        # Sparse Selection
        selection = self.sparse_selector(importance, uncertainty, coarse_logits)

        return {
            "coarse": coarse_logits,
            "importance": selection["importance"],
            "uncertainty": selection["uncertainty"],
            "focus": selection["focus"],
            "patch_coords": selection["patch_coords"],
            "selected_patches": selection["selected_patches"],   # will be replaced with real patches later
            "final": coarse_logits  # temporary, will be fused later
        }


# Quick Test
if __name__ == "__main__":
    module = RefinementModule(in_channels=64, top_k=6, patch_size=64)
    features = torch.randn(2, 64, 192, 192)

    out = module(features)

    print("✅ RefinementModule Test Passed!")
    print(f"Coarse shape       : {out['coarse'].shape}")
    print(f"Number of patches  : {len(out['patch_coords'])}")
    print(f"Example coord      : {out['patch_coords'][0] if out['patch_coords'] else None}")