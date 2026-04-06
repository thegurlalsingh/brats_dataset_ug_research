import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. PATCHIFY / UNPATCHIFY
# =========================================================
def patchify(x, patch_size):
    """x: (B, C, H, W) → (B, N, C, p, p)"""
    B, C, H, W = x.shape
    p = patch_size

    pad_h = (p - H % p) % p
    pad_w = (p - W % p) % p
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))

    B, C, H, W = x.shape
    x = x.view(B, C, H // p, p, W // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = x.view(B, -1, C, p, p)
    return patches  # (B, N, C, p, p)


def unpatchify(patches, patch_size, original_H, original_W):
    """(B, N, C, p, p) → (B, C, H, W)"""
    B, N, C, p, _ = patches.shape
    h = original_H // patch_size
    w = original_W // patch_size
    patches = patches.view(B, h, w, C, p, p)
    patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    return patches.view(B, C, original_H, original_W)


# =========================================================
# 2. PATCH SCORING
# =========================================================
def patch_scores(focus_map, patch_size):
    """focus_map: (B, 1, H, W) → scores: (B, N)"""
    patches = patchify(focus_map, patch_size)   # (B, N, 1, p, p)
    scores = patches.mean(dim=[2, 3, 4])        # (B, N)
    return scores


# =========================================================
# 3. TOP-K SELECTION (fully batched, no Python loop)
# =========================================================
def select_topk_patches(patches, scores, k):
    """
    FIX: Takes a single int k (not per-sample list).
    Fully batched with torch.topk — no Python loop over batch.

    patches : (B, N, C, p, p)
    scores  : (B, N)
    k       : int
    Returns : selected (B, k, C, p, p), indices (B, k)
    """
    B, N, C, p, _ = patches.shape

    # (B, k) indices of top-k patches per sample
    topk_vals, topk_idx = torch.topk(scores, k, dim=1)   # (B, k)

    # Gather: expand indices to match patch dims
    idx_exp = topk_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, k, 1, 1, 1)
    idx_exp = idx_exp.expand(-1, -1, C, p, p)                     # (B, k, C, p, p)

    selected = torch.gather(patches, 1, idx_exp)   # (B, k, C, p, p)

    return selected, topk_idx


# =========================================================
# 4. TOKENS
# =========================================================
def patches_to_tokens(patches):
    """(B, k, C, p, p) → (B, k, C*p*p)"""
    B, k, C, p, _ = patches.shape
    return patches.reshape(B, k, C * p * p)


def tokens_to_patches(tokens, C, p):
    """(B, k, C*p*p) → (B, k, C, p, p)"""
    B, k, _ = tokens.shape
    return tokens.reshape(B, k, C, p, p)


# =========================================================
# 5. TOKEN TRANSFORMER
# =========================================================
class TokenTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            batch_first=True,
            norm_first=True,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)


# =========================================================
# 6. VOLUME AGGREGATOR
# =========================================================
class VolumeAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        # x: (B, S, 1, H, W) or (B, S, H, W)
        if x.dim() == 5:
            x = x.squeeze(2)          # (B, S, H, W)
        x = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        w = self.attn(x)
        x = x * w
        return self.conv(x)           # (B, 1, H, W)


# =========================================================
# 7. LEARNABLE K
# =========================================================
class LearnableK(nn.Module):
    """
    FIX: Now returns a single int k for the whole batch (not per-sample tensor).
    This enables the fully-batched topk selection above.
    max_ratio: maximum fraction of total patches to select.
    """
    def __init__(self, in_channels, max_ratio=0.15):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1)
        self.max_ratio = max_ratio

    def forward(self, x, total_patches):
        B, C, _, _ = x.shape
        g = self.pool(x).view(B, C)
        # Average ratio across batch → single k for whole batch
        ratio = torch.sigmoid(self.fc(g)).mean() * self.max_ratio
        k = max(1, int((ratio * total_patches).item()))
        k = min(k, total_patches)
        return k


# =========================================================
# 8. SCATTER BACK (batched, no Python loop)
# =========================================================
def scatter_patches(patches, indices, original_shape, patch_size):
    """
    FIX: Fully batched scatter — no Python loop over batch dimension.

    patches  : (B, k, C, p, p)
    indices  : (B, k)  — all valid (no -1 padding needed anymore)
    original_shape: (B, C, H, W)
    """
    B, C, H, W = original_shape
    p = patch_size
    h = H // p
    w = W // p
    N = h * w

    # Start with zeros
    full = torch.zeros(B, N, C, p, p, device=patches.device)

    # Expand indices for scatter
    k = patches.shape[1]
    idx_exp = indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, k, 1, 1, 1)
    idx_exp = idx_exp.expand(-1, -1, C, p, p)                    # (B, k, C, p, p)

    full.scatter_(1, idx_exp, patches)

    return unpatchify(full, p, H, W)


# =========================================================
# 9. FULL SPARSE SELECTION MODULE
# =========================================================
class SparseSelectionModule(nn.Module):
    def __init__(self, in_channels=64, patch_size=8, num_slices=5):
        super().__init__()
        self.patch_size = patch_size
        self.token_dim = in_channels * patch_size * patch_size
        self.in_channels = in_channels

        self.token_refiner = TokenTransformer(self.token_dim, num_layers=1)
        self.learnable_k = LearnableK(in_channels)
        self.volume_agg = VolumeAggregator()

    def forward(self, features, focus_stack):
        """
        features   : (B, C, H, W)
        focus_stack: (B, S, 1, H, W)
        """
        B, C, H, W = features.shape

        # 1. Aggregate focus stack → (B, 1, H, W)
        focus_map = self.volume_agg(focus_stack)

        # Ensure focus_map matches feature spatial size
        if focus_map.shape[-2:] != (H, W):
            focus_map = F.interpolate(focus_map, size=(H, W), mode='bilinear', align_corners=False)

        # 2. Patchify features
        patches = patchify(features, self.patch_size)   # (B, N, C, p, p)
        total_patches = patches.shape[1]

        # 3. Score patches by focus map
        scores = patch_scores(focus_map, self.patch_size)   # (B, N)

        # 4. Learnable K — single int for whole batch
        k = self.learnable_k(features, total_patches)

        # 5. Select top-k patches (fully batched)
        selected_patches, selected_indices = select_topk_patches(patches, scores, k)
        # selected_patches: (B, k, C, p, p)
        # selected_indices: (B, k)

        # 6. Tokenize + refine
        tokens = patches_to_tokens(selected_patches)         # (B, k, C*p*p)
        tokens = self.token_refiner(tokens)                  # (B, k, C*p*p)

        # 7. Back to patches
        refined_patches = tokens_to_patches(tokens, C, self.patch_size)  # (B, k, C, p, p)

        # 8. Scatter refined patches back to full map
        refined_map = scatter_patches(
            refined_patches, selected_indices, (B, C, H, W), self.patch_size
        )

        # 9. Residual: keep original features where not selected
        #    Build a mask of which patches were updated
        with torch.no_grad():
            p = self.patch_size
            h, w = H // p, W // p
            N = h * w
            mask = torch.zeros(B, N, device=features.device)
            mask.scatter_(1, selected_indices, 1.0)
            mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1, 1)
            mask = mask.expand(-1, -1, C, p, p)
            full_mask = unpatchify(mask, p, H, W)   # (B, C, H, W)

        # Where mask=1 use refined, where mask=0 keep original
        out = refined_map * full_mask + features * (1 - full_mask)

        return out


# =========================================================
# Quick Test
# =========================================================
if __name__ == "__main__":
    module = SparseSelectionModule(in_channels=64, patch_size=8, num_slices=3)
    features = torch.randn(2, 64, 192, 192)
    focus_stack = torch.randn(2, 3, 1, 192, 192)

    out = module(features, focus_stack)

    print("SparseSelectionModule Test Successful!")
    print(f"Output shape     : {out.shape}")
    print(f"Total parameters : {sum(p.numel() for p in module.parameters()):,}")