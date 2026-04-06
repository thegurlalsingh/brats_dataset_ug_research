import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ===================================================================
# 1. Residual CNN Block
# ===================================================================
class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


# ===================================================================
# 2. Learnable Slice Weighting
# ===================================================================
class LearnableSliceWeighting(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.weights = nn.Parameter(torch.ones(k))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # x: (B, k, C, H, W)
        weights = self.softmax(self.weights).view(1, self.k, 1, 1, 1)
        return x * weights


# ===================================================================
# 3. Cross-Slice Attention
# ===================================================================
class ImprovedCrossSliceAttention(nn.Module):
    def __init__(self, channels=64, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: (B, k, C, H, W)
        FIX: downsample spatial before attention, upsample after.
        Original ran attention at full 192x192 resolution per pixel → 36864 tokens per item.
        Now runs at 24x24 → 576 tokens. ~64x fewer operations.
        """
        B, k, C, H, W = x.shape
        center_idx = k // 2

        # FIX: Downsample spatial dims before attention (192→24)
        scale = 8
        x_down = x.view(B * k, C, H, W)
        x_down = F.avg_pool2d(x_down, kernel_size=scale, stride=scale)
        _, _, h, w = x_down.shape
        x_down = x_down.view(B, k, C, h, w)

        # (B, k, C, h, w) → (B, h, w, k, C)
        x_seq = x_down.permute(0, 3, 4, 1, 2)
        # flatten spatial → (B*h*w, k, C)
        x_seq = x_seq.reshape(B * h * w, k, C)

        # Query = center slice, Key/Value = all slices
        query = x_seq[:, center_idx:center_idx + 1, :]
        key = value = x_seq

        attn_out, _ = self.attn(query, key, value)

        # Reshape back → (B, C, h, w)
        attn_out = attn_out.reshape(B, h, w, C).permute(0, 3, 1, 2)

        # Upsample back to original resolution
        attn_out = F.interpolate(attn_out, size=(H, W), mode='bilinear', align_corners=False)

        # Residual with center slice (full resolution)
        center = x[:, center_idx]   # (B, C, H, W)
        return center + self.gamma * attn_out


# ===================================================================
# 4. Spatial-Reduced Transformer
# ===================================================================
class SpatialReducedTransformer(nn.Module):
    """
    FIX: Replaces the original transformer that ran on 192*192=36864 tokens.
    Uses strided conv to reduce spatial to 24x24=576 tokens before transformer,
    then upsamples back. Makes transformer tractable on full-res feature maps.
    """
    def __init__(self, channels=64, spatial_size=192, reduction=8, num_layers=2):
        super().__init__()
        self.reduction = reduction
        reduced = spatial_size // reduction  # 192 // 8 = 24

        self.down = nn.Conv2d(channels, channels, kernel_size=reduction, stride=reduction)
        self.up = nn.Upsample(size=(spatial_size, spatial_size), mode='bilinear', align_corners=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=4,
            dim_feedforward=channels * 2,
            activation='gelu',
            batch_first=True,
            norm_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Downsample spatial
        x_small = self.down(x)                               # (B, C, h, w)
        _, _, h, w = x_small.shape

        # Flatten to sequence
        tokens = x_small.flatten(2).permute(0, 2, 1)        # (B, h*w, C)
        tokens = self.transformer(tokens)                    # (B, h*w, C)

        # Back to spatial
        x_small = tokens.permute(0, 2, 1).view(B, C, h, w)  # (B, C, h, w)

        # Upsample back
        out = self.up(x_small)                               # (B, C, H, W)

        # Residual
        return x + out


# ===================================================================
# 5. Hybrid 2.5D Backbone
# ===================================================================
class Hybrid2_5DBackbone(nn.Module):
    def __init__(self, k=3, base_channels=64):
        super().__init__()
        self.k = k

        self.cnn_block1 = ResidualCNNBlock(4, base_channels)
        self.cnn_block2 = ResidualCNNBlock(base_channels, base_channels)

        self.slice_weighting = LearnableSliceWeighting(k=k)
        self.cross_attn = ImprovedCrossSliceAttention(channels=base_channels)

        # FIX: Use spatial-reduced transformer instead of full-resolution one
        self.transformer = SpatialReducedTransformer(
            channels=base_channels,
            spatial_size=192,
            reduction=8,
            num_layers=2
        )

    def forward(self, x):
        B, _, H, W = x.shape
        k = self.k

        # (B, 4*k, H, W) → (B, k, 4, H, W)
        x = x.view(B, k, 4, H, W)

        # Per-slice CNN (use gradient checkpointing to save memory)
        features = []
        for i in range(k):
            feat = checkpoint(self.cnn_block1, x[:, i], use_reentrant=False)
            feat = checkpoint(self.cnn_block2, feat, use_reentrant=False)
            features.append(feat)

        x = torch.stack(features, dim=1)   # (B, k, 64, H, W)

        # Learnable slice weighting
        x = self.slice_weighting(x)

        # Cross-slice attention (now runs at downsampled resolution)
        center_features = self.cross_attn(x)   # (B, 64, H, W)

        # Spatial-reduced global transformer
        out = self.transformer(center_features)   # (B, 64, H, W)

        return out


# ===================================================================
# 6. Full Model
# ===================================================================
class SparseBraTSModel(nn.Module):
    def __init__(self, k=3, num_classes=3):
        super().__init__()
        self.backbone = Hybrid2_5DBackbone(k=k, base_channels=64)

        self.coarse_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

        self.importance_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        features = self.backbone(x)

        return {
            'coarse': self.coarse_head(features),
            'importance': self.importance_head(features),
            'uncertainty': self.uncertainty_head(features),
            'features': features
        }


# ===================================================================
# Quick Test
# ===================================================================
if __name__ == "__main__":
    model = SparseBraTSModel(k=3)
    dummy = torch.randn(2, 12, 192, 192)
    out = model(dummy)

    print("Model Test Successful!")
    print(f"Coarse shape      : {out['coarse'].shape}")
    print(f"Importance shape  : {out['importance'].shape}")
    print(f"Uncertainty shape : {out['uncertainty'].shape}")
    print(f"Features shape    : {out['features'].shape}")
    print(f"Total parameters  : {sum(p.numel() for p in model.parameters()):,}")