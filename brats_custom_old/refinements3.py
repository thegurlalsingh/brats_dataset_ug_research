import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# 🔹 1. Coarse Segmentation Head
# -------------------------------
class CoarseHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        return self.net(x)  # logits


# -------------------------------
# 🔹 2. Importance Head
# -------------------------------
class ImportanceHead(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))  # (B,1,H,W)


# -------------------------------
# 🔹 3. Uncertainty Computation
# -------------------------------
def compute_uncertainty(logits):
    """
    logits: (B, C, H, W)
    """
    probs = torch.sigmoid(logits)

    # p * (1 - p)
    uncertainty = probs * (1 - probs)

    # reduce across classes → max uncertainty
    uncertainty, _ = torch.max(uncertainty, dim=1, keepdim=True)

    return uncertainty  # (B,1,H,W)


# -------------------------------
# 🔹 4. Top-K Sparse Selection
# -------------------------------
def topk_mask(focus_map, k_ratio=0.1):
    """
    Returns:
        mask: (B,1,H,W)
        coords: list of tensors [(K,2), ...]
    """
    B, _, H, W = focus_map.shape
    K = int(H * W * k_ratio)

    mask = torch.zeros_like(focus_map)
    coords_list = []

    for b in range(B):
        flat = focus_map[b].view(-1)

        _, topk_idx = torch.topk(flat, K)

        mask[b].view(-1)[topk_idx] = 1.0

        # 🔥 convert indices → (y, x)
        y = topk_idx // W
        x = topk_idx % W
        coords = torch.stack([y, x], dim=1)  # (K,2)

        coords_list.append(coords)

    return mask, coords_list


# -------------------------------
# 🔹 5. Mini Refiner (light CNN)
# -------------------------------
class MiniRefiner(nn.Module):
    def __init__(self, in_channels=64, num_classes=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        return self.net(x)
    

def extract_patches(features, coords_list, patch_size=64):
    """
    features: (B, C, H, W)
    coords_list: list of (K,2)
    returns: (B, K, C, P, P)
    """
    B, C, H, W = features.shape
    P = patch_size
    half = P // 2

    patches_batch = []

    for b in range(B):
        coords = coords_list[b]
        patches = []

        for (y, x) in coords:
            y, x = int(y), int(x)

            y1, y2 = y - half, y + half
            x1, x2 = x - half, x + half

            # padding if needed
            pad_y1 = max(0, -y1)
            pad_y2 = max(0, y2 - H)
            pad_x1 = max(0, -x1)
            pad_x2 = max(0, x2 - W)

            patch = features[b:b+1, :, max(0,y1):min(H,y2), max(0,x1):min(W,x2)]

            patch = F.pad(patch, (pad_x1, pad_x2, pad_y1, pad_y2))

            patches.append(patch.squeeze(0))  # (C,P,P)

        patches = torch.stack(patches)  # (K,C,P,P)
        patches_batch.append(patches)

    return torch.stack(patches_batch)  # (B,K,C,P,P)


# -------------------------------
# 🔹 6. Full Refinement Module
# -------------------------------
class RefinementModule(nn.Module):
    def __init__(self, in_channels=64, num_classes=3, k_ratio=0.1):
        super().__init__()

        self.coarse_head = CoarseHead(in_channels, num_classes)
        self.importance_head = ImportanceHead(in_channels)
        self.refiner = MiniRefiner(in_channels, num_classes)

        self.k_ratio = k_ratio

    def forward(self, features):
        """
        features: (B, C, H, W)
        """

        # 🔹 1. Coarse prediction
        coarse_logits = self.coarse_head(features)

        # 🔹 2. Importance map
        importance = self.importance_head(features)

        # 🔹 3. Uncertainty map
        uncertainty = compute_uncertainty(coarse_logits)

        # 🔹 4. Focus map (YOUR CORE IDEA 🔥)
        focus = importance * uncertainty

        # 🔹 5. Top-K Sparse Mask
        sparse_mask, patch_coords = topk_mask(focus, self.k_ratio)

        # 🔹 6. Apply mask (only refine important + uncertain regions)
        refined_features = features * sparse_mask

        # 🔹 7. Refinement
        refined_logits = self.refiner(refined_features)

        # 🔹 8. Fuse coarse + refined
        final_logits = coarse_logits + refined_logits

        selected_patches = extract_patches(features, patch_coords, patch_size=64)

        return {
            "final": final_logits,
            "coarse": coarse_logits,
            "importance": importance,
            "uncertainty": uncertainty,
            "focus": focus,
            "mask": sparse_mask,
            "patch_coords": patch_coords,
            "selected_patches": selected_patches
        }