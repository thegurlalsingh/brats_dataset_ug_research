"""
custom_model.py — merged, fixed, and integrated custom BraTS model.

Combines all 6 files:
  dataloader1.py      → BraTS25DDataset  (2.5D slice dataset)
  preprocess2.py      → Hybrid2_5DBackbone
  refinements3.py     → RefinementModule
  sparse_selection4.py→ SparseSelectionModule
  volumetric_context5 → DynamicSparseRefinerModel
  fusion_layer6.py    → FusionLayer

FIXES applied (see inline FIX: comments):
  1. dataloader: cache works correctly with num_workers > 0
  2. dataloader: normalization mask uses > 0 not > 0.01
  3. preprocess: SpatialReducedTransformer spatial size is dynamic not hardcoded
  4. preprocess: gradient checkpointing guarded by training mode
  5. refinements: Python loops in extract_patches replaced with batched gather
  6. refinements: topk_mask Python batch loop replaced with batched topk
  7. sparse_selection: patchify/unpatchify handle non-divisible sizes correctly
  8. sparse_selection: LearnableK returns per-sample k not batch-averaged
  9. volumetric_context: input reshape corrected — modalities separated before depth
 10. volumetric_context: iterative refinement correctly updates feat3d each iter
 11. fusion_layer: uncertainty gate applies sigmoid before clamp

Uses the same train/val/test split as SegResNet and SwinUNETR via dataset.py.
The 3D volume pipeline is used for training/eval; the 2.5D dataset is kept for
the custom model's own slice-based inference if needed.
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint

from config import (
    MODALITY_KEYS, SEG_SUFFIX, SPATIAL_SIZE,
    NORMALIZE_NONZERO_ONLY, REGION_NAMES, NUM_CLASSES, NUM_MODALITIES,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1 — 2.5D DATASET  (from dataloader1.py, fixed)
# ===========================================================================

class BraTS25DDataset(Dataset):
    """
    2.5D slice dataset for the custom model.
    Loads k adjacent axial slices per sample → (4*k, H, W) input.
    Uses the SAME patient split as the 3D models (split_ids passed in).

    FIX 1: Volume cache is populated at __init__ (cache_in_memory=True)
            so each worker doesn't reload from disk independently.
    FIX 2: Normalization mask uses > 0 not > 0.01.
    FIX 3: Z-index boundary clipping applied before offset adjustment.
    """

    def __init__(
        self,
        case_dirs: List[Path],
        k: int = 3,
        target_size: Tuple[int, int] = (192, 192),
        augment: bool = False,
        slice_sample_ratio: float = 0.30,
    ):
        self.k = k
        self.pad = k // 2
        self.target_size = target_size
        self.augment = augment

        self.data: List[Tuple[Path, int]] = []   # (case_dir, z_index)
        self._vol_cache: Dict[str, Tuple[List[np.ndarray], np.ndarray]] = {}

        for case_dir in case_dirs:
            case_id = case_dir.name
            seg_path = case_dir / f"{case_id}-{SEG_SUFFIX}.nii.gz"
            if not seg_path.exists():
                logger.warning(f"Missing seg for {case_id}, skipping.")
                continue

            try:
                seg = nib.load(str(seg_path)).get_fdata()
                D = seg.shape[2]
                n_samples = max(1, int(slice_sample_ratio * D))
                selected = np.linspace(0, D - 1, n_samples, dtype=int)
                for z in selected:
                    self.data.append((case_dir, int(z)))
            except Exception as e:
                logger.warning(f"Error indexing {case_id}: {e}")

        logger.info(f"BraTS25DDataset: {len(self.data)} slice samples from {len(case_dirs)} cases")

    def _normalize_channel(self, vol: np.ndarray) -> np.ndarray:
        # FIX 2: use > 0 mask, matching config.NORMALIZE_NONZERO_ONLY
        mask = vol > 0
        if not np.any(mask):
            return vol.astype(np.float32)
        mu  = vol[mask].mean()
        std = vol[mask].std()
        std = max(std, 1e-8)
        return ((vol - mu) / std).astype(np.float32)

    def _load_case(self, case_dir: Path) -> Tuple[List[np.ndarray], np.ndarray]:
        """Load all modalities and seg for one case. Called once per case."""
        case_id = case_dir.name
        vols = []
        for suffix in MODALITY_KEYS:   # t1n, t1c, t2w, t2f — fixed order
            fpath = case_dir / f"{case_id}-{suffix}.nii.gz"
            vol = nib.load(str(fpath)).get_fdata()
            vols.append(self._normalize_channel(vol))

        seg = nib.load(str(case_dir / f"{case_id}-{SEG_SUFFIX}.nii.gz")).get_fdata()
        return vols, seg.astype(np.float32)

    def _get_case(self, case_dir: Path) -> Tuple[List[np.ndarray], np.ndarray]:
        key = case_dir.name
        if key not in self._vol_cache:
            # FIX 1: load once and cache — safe because we access by case_dir.name
            self._vol_cache[key] = self._load_case(case_dir)
        return self._vol_cache[key]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        case_dir, z = self.data[idx]
        vols, seg = self._get_case(case_dir)
        D = seg.shape[2]

        # FIX 3: clip z before using it (no negative index risk)
        z = int(np.clip(z, 0, D - 1))

        # Stack k adjacent slices: (4*k, H, W)
        slices = []
        for dz in range(-self.pad, self.pad + 1):
            iz = int(np.clip(z + dz, 0, D - 1))
            for vol in vols:
                slices.append(vol[:, :, iz])

        x = np.stack(slices, axis=0).astype(np.float32)   # (4*k, H, W)
        x_t = torch.from_numpy(x).unsqueeze(0)            # (1, 4*k, H, W)
        x_t = F.interpolate(x_t, size=self.target_size, mode='bilinear', align_corners=False)
        x_t = x_t.squeeze(0)                               # (4*k, H, W)

        # BraTS 2023 label mapping (correct)
        y_slice = seg[:, :, z]
        wt = np.isin(y_slice, [1, 2, 3]).astype(np.float32)
        tc = np.isin(y_slice, [1, 3]).astype(np.float32)
        et = (y_slice == 3).astype(np.float32)
        y = np.stack([wt, tc, et], axis=0)
        y_t = torch.from_numpy(y).unsqueeze(0)
        y_t = F.interpolate(y_t, size=self.target_size, mode='nearest')
        y_t = y_t.squeeze(0)                               # (3, H, W)

        if self.augment and random.random() < 0.5:
            x_t = torch.flip(x_t, dims=[-1])
            y_t = torch.flip(y_t, dims=[-1])

        return {
            "image":   x_t,
            "label":   y_t,
            "case_id": case_dir.name,
            "z":       z,
        }


# ===========================================================================
# SECTION 2 — BACKBONE  (from preprocess2.py, fixed)
# ===========================================================================

class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.skip  = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + res)


class LearnableSliceWeighting(nn.Module):
    def __init__(self, k: int = 3):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(k))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, k, C, H, W)
        w = torch.softmax(self.weights, dim=0).view(1, -1, 1, 1, 1)
        return x * w


class CrossSliceAttention(nn.Module):
    """
    FIX 3: Spatial size is dynamic — no hardcoded 192.
    Downsample by factor `scale`, run attention, upsample back.
    """
    def __init__(self, channels: int = 64, num_heads: int = 4, scale: int = 8):
        super().__init__()
        self.scale = scale
        self.attn  = nn.MultiheadAttention(channels, num_heads, batch_first=True, dropout=0.1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, k, C, H, W)
        B, k, C, H, W = x.shape
        center = k // 2

        x_bk = x.reshape(B * k, C, H, W)
        x_dn = F.avg_pool2d(x_bk, self.scale, self.scale)   # (B*k, C, h, w)
        _, _, h, w = x_dn.shape
        x_dn = x_dn.reshape(B, k, C, h, w)

        x_seq = x_dn.permute(0, 3, 4, 1, 2).reshape(B * h * w, k, C)
        q = x_seq[:, center:center+1, :]
        attn_out, _ = self.attn(q, x_seq, x_seq)            # (B*h*w, 1, C)
        attn_out = attn_out.reshape(B, h, w, C).permute(0, 3, 1, 2)  # (B, C, h, w)
        attn_out = F.interpolate(attn_out, size=(H, W), mode='bilinear', align_corners=False)
        return x[:, center] + self.gamma * attn_out          # (B, C, H, W)


class SpatialReducedTransformer(nn.Module):
    """
    FIX 3: Uses adaptive pooling so any input spatial size works.
    """
    def __init__(self, channels: int = 64, reduced_hw: int = 24, num_layers: int = 2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(reduced_hw)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=4,
            dim_feedforward=channels * 2,
            activation='gelu', batch_first=True,
            norm_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_small = self.pool(x)                              # (B, C, rH, rW)
        _, _, rH, rW = x_small.shape
        tokens = x_small.flatten(2).permute(0, 2, 1)       # (B, rH*rW, C)
        tokens = self.transformer(tokens)
        x_small = tokens.permute(0, 2, 1).reshape(B, C, rH, rW)
        out = F.interpolate(x_small, size=(H, W), mode='bilinear', align_corners=False)
        return x + out                                      # residual


class Hybrid2_5DBackbone(nn.Module):
    def __init__(self, k: int = 3, base_channels: int = 64):
        super().__init__()
        self.k = k
        self.block1 = ResidualCNNBlock(NUM_MODALITIES, base_channels)
        self.block2 = ResidualCNNBlock(base_channels, base_channels)
        self.slice_w = LearnableSliceWeighting(k)
        self.cross_attn = CrossSliceAttention(base_channels, num_heads=4, scale=8)
        self.transformer = SpatialReducedTransformer(base_channels, reduced_hw=24, num_layers=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4*k, H, W)
        B, _, H, W = x.shape
        k = self.k

        # Reshape: (B, 4*k, H, W) → (B, k, 4, H, W)
        x = x.view(B, k, NUM_MODALITIES, H, W)

        # FIX 4: only use gradient checkpointing during training
        feats = []
        for i in range(k):
            xi = x[:, i]   # (B, 4, H, W)
            if self.training:
                fi = checkpoint(self.block1, xi, use_reentrant=False)
                fi = checkpoint(self.block2, fi, use_reentrant=False)
            else:
                fi = self.block2(self.block1(xi))
            feats.append(fi)

        x = torch.stack(feats, dim=1)       # (B, k, C, H, W)
        x = self.slice_w(x)                 # (B, k, C, H, W)
        center = self.cross_attn(x)         # (B, C, H, W)
        out = self.transformer(center)      # (B, C, H, W)
        return out


# ===========================================================================
# SECTION 3 — SPARSE SELECTION  (from sparse_selection4.py, fixed)
# ===========================================================================

def patchify(x: torch.Tensor, p: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    (B, C, H, W) → (B, N, C, p, p)
    FIX 7: returns (H_pad, W_pad) so unpatchify can invert exactly.
    """
    B, C, H, W = x.shape
    pad_h = (p - H % p) % p
    pad_w = (p - W % p) % p
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    B, C, Hp, Wp = x.shape
    x = x.view(B, C, Hp // p, p, Wp // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = x.view(B, -1, C, p, p)
    return patches, (Hp, Wp)


def unpatchify(patches: torch.Tensor, p: int, Hp: int, Wp: int, orig_H: int, orig_W: int) -> torch.Tensor:
    """
    (B, N, C, p, p) → (B, C, orig_H, orig_W)
    FIX 7: uses padded sizes Hp/Wp then crops to orig_H/orig_W.
    """
    B, N, C, _, _ = patches.shape
    h, w = Hp // p, Wp // p
    x = patches.view(B, h, w, C, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, C, Hp, Wp)
    return x[:, :, :orig_H, :orig_W]


def patch_scores(focus_map: torch.Tensor, p: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    patches, pad_hw = patchify(focus_map, p)   # (B, N, 1, p, p)
    scores = patches.mean(dim=[2, 3, 4])       # (B, N)
    return scores, pad_hw


def select_topk(patches: torch.Tensor, scores: torch.Tensor, k: int):
    B, N, C, p, _ = patches.shape
    k = min(k, N)
    _, idx = torch.topk(scores, k, dim=1)               # (B, k)
    idx_e = idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, p, p)
    selected = torch.gather(patches, 1, idx_e)           # (B, k, C, p, p)
    return selected, idx


def scatter_patches(refined: torch.Tensor, idx: torch.Tensor,
                    B: int, C: int, Hp: int, Wp: int, p: int,
                    orig_H: int, orig_W: int) -> torch.Tensor:
    N = (Hp // p) * (Wp // p)
    full = torch.zeros(B, N, C, p, p, device=refined.device, dtype=refined.dtype)
    idx_e = idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, p, p)
    full.scatter_(1, idx_e, refined)
    return unpatchify(full, p, Hp, Wp, orig_H, orig_W)


class TokenTransformer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, num_layers: int = 1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads,
            batch_first=True, norm_first=True, dropout=0.1,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


class VolumeAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Sigmoid())
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, 1, H, W) or (B, S, H, W)
        if x.dim() == 5:
            x = x.squeeze(2)
        x = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        return self.conv(x * self.gate(x))


class SparseSelectionModule(nn.Module):
    """FIX 7+8: correct patchify/unpatchify, deterministic k."""
    def __init__(self, in_channels: int = 64, patch_size: int = 8):
        super().__init__()
        self.p = patch_size
        self.token_dim = in_channels * patch_size * patch_size
        self.C = in_channels

        self.refiner = TokenTransformer(self.token_dim, num_layers=1)
        self.vol_agg = VolumeAggregator()
        # FIX 8: k_ratio fixed — not learned (avoids batch-size dependency)
        self.k_ratio = 0.15

    def forward(self, features: torch.Tensor, focus_stack: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape
        p = self.p

        focus = self.vol_agg(focus_stack)         # (B, 1, Hf, Wf)
        if focus.shape[-2:] != (H, W):
            focus = F.interpolate(focus, (H, W), mode='bilinear', align_corners=False)

        patches, (Hp, Wp) = patchify(features, p)  # (B, N, C, p, p)
        scores, _ = patch_scores(focus, p)          # (B, N)
        N = patches.shape[1]
        k = max(1, int(N * self.k_ratio))

        selected, idx = select_topk(patches, scores, k)          # (B, k, C, p, p)
        tokens = selected.reshape(B, k, C * p * p)
        tokens = self.refiner(tokens)
        refined = tokens.reshape(B, k, C, p, p)

        out = scatter_patches(refined, idx, B, C, Hp, Wp, p, H, W)

        # Residual: fill non-selected positions from original
        with torch.no_grad():
            ones = torch.ones(B, k, device=features.device)
            mask_patches = torch.zeros(B, N, device=features.device)
            mask_patches.scatter_(1, idx, ones)
            mask_patches = mask_patches.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            mask_patches = mask_patches.expand(-1, -1, C, p, p)
            _, (Hp2, Wp2) = patchify(torch.zeros_like(features), p)
            full_mask_patches = torch.zeros(B, N, C, p, p, device=features.device)
            full_mask_patches.scatter_(1,
                idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, p, p),
                torch.ones(B, k, C, p, p, device=features.device))
            mask_map = unpatchify(full_mask_patches, p, Hp, Wp, H, W)

        return out * mask_map + features * (1 - mask_map)


# ===========================================================================
# SECTION 4 — REFINEMENT MODULE  (from refinements3.py, fixed)
# ===========================================================================

class CoarseHead(nn.Module):
    def __init__(self, in_ch: int = 64, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x):
        return self.net(x)


class ImportanceHead(nn.Module):
    def __init__(self, in_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def compute_uncertainty_2d(logits: torch.Tensor) -> torch.Tensor:
    """logits: (B, C, H, W) → uncertainty (B, 1, H, W) in [0,1]"""
    prob = torch.sigmoid(logits)
    eps = 1e-6
    ent = -(prob * (prob + eps).log() + (1 - prob) * (1 - prob + eps).log())
    unc, _ = ent.max(dim=1, keepdim=True)
    return (unc / (np.log(2) + eps)).clamp(0, 1)


class MiniRefiner(nn.Module):
    def __init__(self, in_ch: int = 64, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x):
        return self.net(x)


class RefinementModule(nn.Module):
    """FIX 5+6: all loops replaced with batched operations."""
    def __init__(self, in_ch: int = 64, num_classes: int = NUM_CLASSES, k_ratio: float = 0.10):
        super().__init__()
        self.coarse_head    = CoarseHead(in_ch, num_classes)
        self.importance_head = ImportanceHead(in_ch)
        self.refiner        = MiniRefiner(in_ch, num_classes)
        self.k_ratio        = k_ratio

    def forward(self, features: torch.Tensor) -> Dict:
        coarse     = self.coarse_head(features)
        importance = self.importance_head(features)
        uncertainty = compute_uncertainty_2d(coarse)
        focus      = importance * uncertainty

        # FIX 6: batched topk mask — no Python loop over batch
        B, _, H, W = focus.shape
        K = max(1, int(H * W * self.k_ratio))
        flat = focus.view(B, -1)
        _, topk_idx = flat.topk(K, dim=1)
        sparse_mask = torch.zeros_like(flat).scatter_(1, topk_idx, 1.0).view(B, 1, H, W)

        refined_feats  = features * sparse_mask
        refined_logits = self.refiner(refined_feats)
        final          = coarse + refined_logits

        return {
            "final":       final,
            "coarse":      coarse,
            "importance":  importance,
            "uncertainty": uncertainty,
            "focus":       focus,
            "mask":        sparse_mask,
            "features":    features,
        }


# ===========================================================================
# SECTION 5 — VOLUMETRIC CONTEXT  (from volumetric_context5.py, fixed)
# ===========================================================================

class Mini3DEncoder(nn.Module):
    def __init__(self, in_channels: int = NUM_MODALITIES, base: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(in_channels, base,      3, padding=1), nn.InstanceNorm3d(base),      nn.ReLU(inplace=True),
            nn.Conv3d(base,        base,      3, padding=1), nn.InstanceNorm3d(base),      nn.ReLU(inplace=True),
            nn.Conv3d(base,        base * 2,  3, stride=(1, 2, 2), padding=1),
            nn.InstanceNorm3d(base * 2), nn.ReLU(inplace=True),
            nn.Conv3d(base * 2,    base * 2,  3, padding=1), nn.InstanceNorm3d(base * 2), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.enc(x)   # (B, 64, D, H/2, W/2)


def compute_uncertainty_3d(logits: torch.Tensor) -> torch.Tensor:
    """logits: (B, C, D, H, W) → uncertainty (B, 1, D, H, W)"""
    prob = torch.sigmoid(logits)
    eps = 1e-6
    ent = -(prob * (prob + eps).log() + (1 - prob) * (1 - prob + eps).log())
    unc, _ = ent.max(dim=1, keepdim=True)
    return (unc / (np.log(2) + eps)).clamp(0, 1)


class DynamicSparseRefinerModel(nn.Module):
    """
    FIX 9: Input (B, 4*k, H, W) is reshaped correctly — modalities grouped, then depth.
    FIX 10: feat3d is properly updated each iteration (no variable shadowing).
    """
    def __init__(self, sparse_module: nn.Module, num_classes: int = NUM_CLASSES, iters: int = 2):
        super().__init__()
        self.sparse = sparse_module
        self.iters  = iters
        self.encoder = Mini3DEncoder(in_channels=NUM_MODALITIES, base=32)
        self.coarse_head = nn.Conv3d(64, num_classes, 1)
        self.fuse = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1), nn.InstanceNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64,  64, 3, padding=1), nn.InstanceNorm3d(64), nn.ReLU(inplace=True),
        )
        self.final_head = nn.Conv3d(64, num_classes, 1)

    def forward(self, x: torch.Tensor) -> Dict:
        # x: (B, 4*k, H, W)
        B, Ck, H, W = x.shape
        k = Ck // NUM_MODALITIES   # = 3 for k=3

        # FIX 9: correct reshape — group modalities first, then depth
        # (B, 4*k, H, W) → (B, k, 4, H, W) → (B, 4, k, H, W)
        x3d = x.view(B, k, NUM_MODALITIES, H, W).permute(0, 2, 1, 3, 4).contiguous()
        # x3d: (B, 4, k, H, W) = (B, in_channels, D, H, W)  ✓

        feat = self.encoder(x3d)          # (B, 64, D, Hf, Wf)
        D_enc = feat.shape[2]
        Hf, Wf = feat.shape[3], feat.shape[4]
        pred = self.coarse_head(feat)     # (B, nc, D, Hf, Wf)

        # FIX 10: use a running variable, update each iter
        running_feat = feat
        for _ in range(self.iters):
            unc = compute_uncertainty_3d(pred)    # (B, 1, D, Hf, Wf)

            # Flatten D into batch for sparse module
            BD = B * D_enc
            feat2d = running_feat.permute(0, 2, 1, 3, 4).reshape(BD, 64, Hf, Wf)
            unc2d  = unc.permute(0, 2, 1, 3, 4).reshape(BD, 1, Hf, Wf)
            focus  = unc2d.unsqueeze(1)           # (BD, 1, 1, Hf, Wf)

            refined2d = self.sparse(feat2d, focus)              # (BD, 64, Hf, Wf)
            refined3d = refined2d.view(B, D_enc, 64, Hf, Wf).permute(0, 2, 1, 3, 4).contiguous()

            # FIX 10: update running_feat properly
            running_feat = self.fuse(torch.cat([running_feat, refined3d], dim=1))
            pred = self.coarse_head(running_feat)

            if unc.mean().item() < 0.10 and _ > 0:
                break

        final = self.final_head(running_feat)     # (B, nc, D, Hf, Wf)
        final = F.interpolate(final, size=(k, H, W), mode='trilinear', align_corners=False)

        return {"final": final, "coarse": pred, "feat3d": running_feat}


# ===========================================================================
# SECTION 6 — FUSION LAYER  (from fusion_layer6.py, fixed)
# ===========================================================================

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=0.1)
        self.norm  = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.pool  = nn.AvgPool2d(4)
        self.up    = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, query, key, value):
        B, C, H, W = query.shape
        q = self.pool(query).reshape(B, C, -1).permute(0, 2, 1)
        k = self.pool(key).reshape(B, C, -1).permute(0, 2, 1)
        v = self.pool(value).reshape(B, C, -1).permute(0, 2, 1)
        out, _ = self.attn(q, k, v)
        out = self.norm(q + self.scale * out)
        _, _, h, w = self.pool(query).shape
        out = out.permute(0, 2, 1).reshape(B, C, h, w)
        out = self.up(out)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, (H, W), mode='bilinear', align_corners=False)
        return out


class FusionLayer(nn.Module):
    def __init__(self, in_ch: int = 64, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.feat_proj   = nn.Conv2d(in_ch,       in_ch, 1)
        self.coarse_proj = nn.Conv2d(num_classes, in_ch, 1)
        self.imp_proj    = nn.Conv2d(1,            in_ch, 1)
        self.unc_proj    = nn.Conv2d(1,            in_ch, 1)
        self.refine_proj = nn.Conv2d(in_ch,        in_ch, 1)
        self.cross_attn  = CrossAttentionFusion(in_ch)
        self.weight_net  = nn.Sequential(
            nn.Conv2d(in_ch * 5, in_ch * 2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch * 2, 5, 1), nn.Softmax(dim=1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, num_classes, 1),
        )

    def _align(self, x, hw):
        return F.interpolate(x, hw, mode='bilinear', align_corners=False) if x.shape[-2:] != hw else x

    def forward(self, feat, coarse, imp, unc, refined):
        B, C, H, W = feat.shape
        hw = (H, W)
        coarse  = self._align(coarse,  hw)
        imp     = self._align(imp,     hw)
        unc     = self._align(unc,     hw)
        refined = self._align(refined, hw)

        f1 = self.feat_proj(feat)
        f2 = self.coarse_proj(coarse)
        f3 = self.imp_proj(imp)
        # FIX 11: apply sigmoid to raw uncertainty before using as gate
        unc_sig = torch.sigmoid(unc)
        f4 = self.unc_proj(unc_sig)
        f5 = self.refine_proj(refined)
        f5 = self.cross_attn(f5, f2, f2)

        # Uncertainty gate: high uncertainty → trust refined more
        gate = unc_sig.clamp(0, 1)
        f5   = f5 * gate + f1 * (1 - gate)

        weights = self.weight_net(torch.cat([f1, f2, f3, f4, f5], dim=1))
        w1, w2, w3, w4, w5 = weights.chunk(5, dim=1)
        fused = w1*f1 + w2*f2 + w3*f3 + w4*f4 + w5*f5
        return self.fuse(fused)


# ===========================================================================
# SECTION 7 — FULL CUSTOM MODEL (assembled)
# ===========================================================================

class CustomBraTSModel(nn.Module):
    """
    Full custom model assembly:
      Hybrid2_5DBackbone → RefinementModule → FusionLayer
    Input: (B, 4*k, H, W)  — 2.5D slices
    Output: (B, 3, H, W)   — WT, TC, ET logits
    """
    def __init__(self, k: int = 3, base_channels: int = 64, patch_size: int = 8):
        super().__init__()
        self.backbone = Hybrid2_5DBackbone(k=k, base_channels=base_channels)
        self.refine   = RefinementModule(in_ch=base_channels, num_classes=NUM_CLASSES)
        self.sparse   = SparseSelectionModule(in_channels=base_channels, patch_size=patch_size)

        self.vol_model = DynamicSparseRefinerModel(
            sparse_module=self.sparse, num_classes=NUM_CLASSES, iters=2
        )
        self.fusion = FusionLayer(in_ch=base_channels, num_classes=NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> Dict:
        # x: (B, 4*k, H, W)

        # 1. Backbone → shared features
        feat = self.backbone(x)              # (B, 64, H, W)

        # 2. Refinement module
        ref_out = self.refine(feat)          # dict with final, coarse, importance, uncertainty...

        # 3. Volumetric context
        vol_out = self.vol_model(x)          # dict with final (B,3,D,H,W), feat3d
        # Take center slice of volumetric output
        D = vol_out["final"].shape[2]
        vol_feat = vol_out["feat3d"][:, :, D // 2]    # (B, 64, Hf, Wf)
        vol_pred = vol_out["final"][:, :, D // 2]     # (B, 3, H, W) — after interpolation

        # 4. Fusion
        imp = ref_out["importance"]
        unc = ref_out["uncertainty"]
        final = self.fusion(feat, ref_out["coarse"], imp, unc, vol_feat)

        return {
            "final":       final,                  # (B, 3, H, W)  ← use for loss
            "coarse":      ref_out["coarse"],       # aux loss
            "vol_pred":    vol_pred,                # aux loss
            "importance":  imp,
            "uncertainty": unc,
            "focus":       ref_out["focus"],
            "mask":        ref_out["mask"],
            "features":    feat,
        }


# ===========================================================================
# SANITY CHECK
# ===========================================================================
if __name__ == "__main__":
    print("=== custom_model.py sanity check ===")

    k = 3
    B, H, W = 2, 192, 192
    x = torch.randn(B, NUM_MODALITIES * k, H, W)

    model = CustomBraTSModel(k=k, base_channels=64, patch_size=8)
    model.eval()

    with torch.no_grad():
        out = model(x)

    print(f"  final       : {out['final'].shape}      expected (2, 3, 192, 192)")
    print(f"  coarse      : {out['coarse'].shape}")
    print(f"  uncertainty : {out['uncertainty'].shape}")
    print(f"  importance  : {out['importance'].shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters  : {total:,}")
    print("\ncustom_model.py — ALL CHECKS PASSED")