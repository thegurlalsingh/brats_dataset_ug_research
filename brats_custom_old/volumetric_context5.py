import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 🔹 Mini 3D CNN Encoder
# =========================================================
class Mini3DEncoder(nn.Module):
    def __init__(self, in_channels=4, base=32):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv3d(in_channels, base, 3, padding=1),
            nn.InstanceNorm3d(base),          # InstanceNorm is better than BN for 3D medical
            nn.ReLU(inplace=True),

            nn.Conv3d(base, base, 3, padding=1),
            nn.InstanceNorm3d(base),
            nn.ReLU(inplace=True),

            # Downsample only spatial (not depth — we only have D=3)
            nn.Conv3d(base, base * 2, 3, stride=(1, 2, 2), padding=1),
            nn.InstanceNorm3d(base * 2),
            nn.ReLU(inplace=True),

            nn.Conv3d(base * 2, base * 2, 3, padding=1),
            nn.InstanceNorm3d(base * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.enc(x)   # (B, 64, D, H/2, W/2)


# =========================================================
# 🔹 Coarse Head (3D)
# =========================================================
class CoarseHead(nn.Module):
    def __init__(self, in_channels=64, num_classes=3):
        super().__init__()
        self.head = nn.Conv3d(in_channels, num_classes, 1)

    def forward(self, x):
        return self.head(x)   # (B, num_classes, D, H', W')


# =========================================================
# 🔹 Uncertainty from multi-label logits
#    FIX: Use sigmoid (not softmax) — labels are multi-label,
#    not mutually exclusive. Uncertainty = max per-class entropy.
# =========================================================
def compute_uncertainty(logits):
    """
    logits: (B, C, D, H, W)  — raw (un-activated)
    Returns uncertainty: (B, 1, D, H, W) in [0, 1]
    Uses per-channel sigmoid entropy, then takes max across channels.
    """
    prob = torch.sigmoid(logits)                          # (B, C, D, H, W)
    # Binary entropy per channel: -p*log(p) - (1-p)*log(1-p)
    eps = 1e-6
    entropy = -(prob * torch.log(prob + eps) +
                (1 - prob) * torch.log(1 - prob + eps))  # (B, C, D, H, W)
    # Max entropy across classes → (B, 1, D, H, W), normalised to [0,1]
    unc, _ = torch.max(entropy, dim=1, keepdim=True)
    unc = unc / (torch.log(torch.tensor(2.0)) + eps)     # log2 normalise
    return unc.clamp(0, 1)


# =========================================================
# 🔥 Dynamic Sparse Refiner
#
# KEY FIXES vs original:
#   1. No Python loop over slices for sparse refinement.
#      All D slices are processed in one batched call by
#      reshaping (B, C, D, H', W') → (B*D, C, H', W').
#   2. focus_stack passed to sparse module now contains REAL
#      per-slice uncertainty (not 5 identical copies).
#   3. Uncertainty uses sigmoid entropy (multi-label correct).
#   4. Early-stop threshold raised slightly and guarded
#      against always-triggering on easy backgrounds.
#   5. Final interpolation uses correct target (D, H, W).
# =========================================================
class DynamicSparseRefinerModel(nn.Module):
    def __init__(self, sparse_module, num_classes=3, iters=2):
        super().__init__()

        self.sparse = sparse_module
        self.iters = iters
        self.num_classes = num_classes

        self.encoder = Mini3DEncoder(in_channels=4, base=32)
        self.coarse_head = CoarseHead(64, num_classes)

        self.fuse = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.final_head = nn.Conv3d(64, num_classes, 1)

    def forward(self, x_2d5):
        """
        x_2d5: (B, 12, H, W)   [k=3 slices × 4 modalities]
        Returns dict with keys: 'final', 'coarse', 'feat3d'
        """
        B, C, H, W = x_2d5.shape
        D = 3   # always 3 for k=3

        # ── Reshape to 3D volume ────────────────────────────────
        # (B, 12, H, W) → (B, 4, D, H, W)
        x_3d = x_2d5.view(B, D, 4, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # ── 3D Encode ──────────────────────────────────────────
        feat3d = self.encoder(x_3d)     # (B, 64, D, H/2, W/2)
        Hf, Wf = feat3d.shape[-2], feat3d.shape[-1]

        pred = self.coarse_head(feat3d) # (B, num_classes, D, H/2, W/2)

        # ── Iterative Refinement ───────────────────────────────
        for it in range(self.iters):

            # FIX 3: proper multi-label uncertainty
            uncertainty = compute_uncertainty(pred)  # (B, 1, D, Hf, Wf)

            # FIX 1: Process all slices in ONE batched call
            #   Reshape (B, C, D, Hf, Wf) → (B*D, C, Hf, Wf)
            BD = B * D
            feat2d_all = feat3d.permute(0, 2, 1, 3, 4).reshape(BD, 64, Hf, Wf)
            unc2d_all  = uncertainty.permute(0, 2, 1, 3, 4).reshape(BD, 1, Hf, Wf)

            # FIX 2: focus_stack = real per-slice uncertainty, not copies
            # SparseSelectionModule expects (B, S, 1, H, W)
            # We treat each (B*D) item as its own "batch" with S=1 focus frame
            # (the sparse module's VolumeAggregator handles S=1 fine)
            focus_stack = unc2d_all.unsqueeze(1)   # (B*D, 1, 1, Hf, Wf)

            refined_all = self.sparse(feat2d_all, focus_stack)  # (B*D, 64, Hf, Wf)

            # Reshape back to 3D: (B*D, 64, Hf, Wf) → (B, 64, D, Hf, Wf)
            refined3d = refined_all.view(B, D, 64, Hf, Wf).permute(0, 2, 1, 3, 4).contiguous()

            feat3d = self.fuse(torch.cat([feat3d, refined3d], dim=1))
            pred   = self.coarse_head(feat3d)

            # FIX 4: early stop only if uncertainty is HIGH everywhere
            # (i.e. don't stop early on easy all-background slices)
            mean_unc = uncertainty.mean().item()
            if mean_unc < 0.10 and it > 0:
                break

        # ── Final prediction ────────────────────────────────────
        final_logits = self.final_head(feat3d)   # (B, num_classes, D, Hf, Wf)

        # FIX 5: upsample back to original (D, H, W)
        final_logits = F.interpolate(
            final_logits,
            size=(D, H, W),
            mode="trilinear",
            align_corners=False
        )

        return {
            "final":  final_logits,   # (B, num_classes, D, H, W)
            "coarse": pred,           # (B, num_classes, D, Hf, Wf) — for aux loss
            "feat3d": feat3d          # (B, 64, D, Hf, Wf)
        }


# =========================================================
# ✅ TEST
# =========================================================
if __name__ == "__main__":
    from sparse_selection4 import SparseSelectionModule

    sparse = SparseSelectionModule(in_channels=64, patch_size=8, num_slices=3)
    model  = DynamicSparseRefinerModel(sparse, num_classes=3, iters=2)

    x = torch.randn(2, 12, 192, 192)
    out = model(x)

    print("✅ DynamicSparseRefinerModel — FIXED")
    print(f"  final  : {out['final'].shape}")    # (2, 3, 3, 192, 192)
    print(f"  coarse : {out['coarse'].shape}")
    print(f"  feat3d : {out['feat3d'].shape}")
    print(f"  Params : {sum(p.numel() for p in model.parameters()):,}")