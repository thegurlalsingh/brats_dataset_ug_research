import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 🔹 Cross Attention Fusion
#    FIX: pool before attention (192 → 48) to avoid OOM
# =========================================================
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=64, num_heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.norm  = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

        self.pool     = nn.AvgPool2d(4)                                          # 192 → 48
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, query, key, value):
        B, C, H, W = query.shape

        q = self.pool(query)
        k = self.pool(key)
        v = self.pool(value)
        _, _, h, w = q.shape

        q = q.reshape(B, C, -1).permute(0, 2, 1)   # (B, h*w, C)
        k = k.reshape(B, C, -1).permute(0, 2, 1)
        v = v.reshape(B, C, -1).permute(0, 2, 1)

        attn_out, _ = self.attn(q, k, v)
        out = self.norm(q + self.scale * attn_out)

        out = out.permute(0, 2, 1).reshape(B, C, h, w)
        out = self.upsample(out)

        # Safety: ensure spatial matches input
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return out


# =========================================================
# 🔥 Fusion Layer
#
# KEY FIXES vs original:
#   1. Uncertainty gate now uses raw uncertainty directly
#      (no double-sigmoid). High uncertainty → trust refined
#      features more; low uncertainty → trust original features.
#   2. Coarse logits projected correctly from num_classes → dim.
#   3. All spatial alignment done safely before cat/addition.
#   4. No sigmoid applied inside uncertainty computation here —
#      that is done in main.py before passing in.
# =========================================================
class FusionLayer(nn.Module):
    def __init__(self, in_channels=64, num_classes=3):
        super().__init__()

        self.feat_proj    = nn.Conv2d(in_channels,  in_channels, 1)
        self.coarse_proj  = nn.Conv2d(num_classes,  in_channels, 1)
        self.imp_proj     = nn.Conv2d(1,             in_channels, 1)
        self.unc_proj     = nn.Conv2d(1,             in_channels, 1)
        self.refine_proj  = nn.Conv2d(in_channels,  in_channels, 1)

        self.cross_attn = CrossAttentionFusion(dim=in_channels)

        # Adaptive weight generator over 5 streams
        self.weight_net = nn.Sequential(
            nn.Conv2d(in_channels * 5, in_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, 5, 1),
            nn.Softmax(dim=1)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1)
        )

    def _align(self, x, target_hw):
        """Safely align spatial dims of x to target_hw if needed."""
        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        return x

    def forward(self, feat, coarse, imp, unc, refined):
        """
        feat    : (B, in_channels, H, W)   — backbone features
        coarse  : (B, num_classes, H, W)   — coarse logits (2D)
        imp     : (B, 1, H, W)             — importance map [0,1]
        unc     : (B, 1, H, W)             — uncertainty map [0,1]
        refined : (B, in_channels, H, W)   — refined features from 3D branch
        """
        B, C, H, W = feat.shape
        target_hw = (H, W)

        # ── Align all inputs to feat spatial size ──────────────
        coarse  = self._align(coarse,  target_hw)
        imp     = self._align(imp,     target_hw)
        unc     = self._align(unc,     target_hw)
        refined = self._align(refined, target_hw)

        # ── Project all streams to in_channels ─────────────────
        f1 = self.feat_proj(feat)           # (B, C, H, W)
        f2 = self.coarse_proj(coarse)       # (B, C, H, W)
        f3 = self.imp_proj(imp)             # (B, C, H, W)
        f4 = self.unc_proj(unc)             # (B, C, H, W)
        f5 = self.refine_proj(refined)      # (B, C, H, W)

        # ── Cross attention: refine features guided by coarse ──
        f5 = self.cross_attn(f5, f2, f2)   # (B, C, H, W)

        # ── FIX 1: Uncertainty-based gate ─────────────────────
        # High uncertainty → gate open → trust refined (f5) more
        # Low uncertainty  → gate closed → trust original (f1)
        gate = unc.clamp(0, 1)             # already in [0,1], no double sigmoid
        f5   = f5 * gate + f1 * (1 - gate)

        # ── Adaptive weighted fusion ───────────────────────────
        combined = torch.cat([f1, f2, f3, f4, f5], dim=1)   # (B, 5C, H, W)
        weights  = self.weight_net(combined)                  # (B, 5, H, W)
        w1, w2, w3, w4, w5 = torch.chunk(weights, 5, dim=1)

        fused = w1*f1 + w2*f2 + w3*f3 + w4*f4 + w5*f5       # (B, C, H, W)

        # ── Final output ───────────────────────────────────────
        out = self.fuse(fused)   # (B, num_classes, H, W)

        return out


# =========================================================
# ✅ TEST
# =========================================================
if __name__ == "__main__":
    fusion = FusionLayer(in_channels=64, num_classes=3)

    feat    = torch.randn(2, 64, 192, 192)
    coarse  = torch.randn(2,  3, 192, 192)
    imp     = torch.rand (2,  1, 192, 192)
    unc     = torch.rand (2,  1, 192, 192)
    refined = torch.randn(2, 64, 192, 192)

    out = fusion(feat, coarse, imp, unc, refined)

    print("✅ FusionLayer Test Successful!")
    print(f"  Output shape : {out.shape}")
    print(f"  Total params : {sum(p.numel() for p in fusion.parameters()):,}")