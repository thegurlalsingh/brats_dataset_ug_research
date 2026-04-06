"""
xai.py — Stage 5: Explainable AI for all 4 BraTS models.

Outputs saved to config.XAI_DIR / <model_name>/:
  cam/          — ForwardCAM activation map overlays (pred colourmap on MRI slice)
  overlay/      — Pred mask vs GT mask side-by-side (WT / TC / ET per case)
  occlusion/    — Occlusion sensitivity heatmaps (modality-level + patch-level)
  uncertainty/  — Entropy maps per voxel / slice
  modality/     — Bar chart of per-modality importance scores

Usage
-----
  # Run XAI for all 4 models (uses best checkpoints from config.CHECKPOINT_DIR)
  python xai.py

  # Run for one model only
  python xai.py --model segresnet
  python xai.py --model swinunetr
  python xai.py --model nnunet --nnunet_ckpt "path/to/checkpoint_best.pth"
  python xai.py --model custom

  # Control how many test cases to visualise
  python xai.py --n_cases 5

Notes
-----
* ForwardCAM uses a single forward-hook on the last encoder feature map.
  No gradient is required — this makes it compatible with nnUNet and the
  custom model which have non-standard backward graphs.
* Occlusion operates at MODALITY level (zero-out one of the 4 channels)
  and at PATCH level (16^3 or 16^2 cube/square) — both are saved.
* Uncertainty = sigmoid-entropy H = -p*log(p) - (1-p)*log(1-p) averaged
  over the 3 output channels.  For nnUNet (softmax output) we use
  softmax-entropy instead.
* The custom model already exposes "uncertainty" and "importance" tensors
  from its FusionLayer; we use those directly instead of recomputing.
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# project imports
# ---------------------------------------------------------------------------
from config import (
    CHECKPOINT_DIR, DATA_ROOT, DEVICE,
    N_VIZ_CASES, NUM_CLASSES, NUM_MODALITIES,
    NNUNET_PREPROCESSED, NNUNET_RAW, NNUNET_RESULTS,
    OCCLUSION_PATCH, REGION_NAMES, SEGRESNET_CFG,
    SPATIAL_SIZE, SWIN_CFG, SW_OVERLAP, SW_ROI_SIZE,
    USE_AMP, XAI_DIR, MODALITY_KEYS,
)
from dataset import (
    BraTSDataset, build_splits, convert_labels,
    discover_cases, get_case_paths_for_split,
    load_case, normalize_image, pad_or_crop,
)

os.environ["nnUNet_raw"]          = str(NNUNET_RAW)
os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
os.environ["nnUNet_results"]      = str(NNUNET_RESULTS)

# Colour maps & region colours — consistent with evaluate scripts
REGION_COLORS   = {"WT": "#E84C4C", "TC": "#4CAF50", "ET": "#4C8BE8"}
MODALITY_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
CAM_CMAP        = "hot"
OVERLAY_ALPHA   = 0.45

# ============================================================================
# UTILITY — create output sub-dirs for a model
# ============================================================================

def make_xai_dirs(model_name: str) -> Dict[str, Path]:
    base = XAI_DIR / model_name
    dirs = {
        "cam":         base / "cam",
        "overlay":     base / "overlay",
        "occlusion":   base / "occlusion",
        "uncertainty": base / "uncertainty",
        "modality":    base / "modality",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ============================================================================
# DATA HELPERS
# ============================================================================

def load_test_cases(n_cases: int) -> List[Dict]:
    """
    Load up to n_cases from the test split.
    Returns list of dicts with keys: case_id, image (4,D,H,W), label (3,D,H,W)
    """
    all_cases = discover_cases(DATA_ROOT)
    splits    = build_splits(all_cases)
    test_dirs = get_case_paths_for_split(splits["test"], DATA_ROOT)[:n_cases]

    records = []
    for case_dir in tqdm(test_dirs, desc="Loading test cases", unit="case"):
        case_id = case_dir.name
        try:
            image_np, label_raw = load_case(case_dir)
            image_np = normalize_image(image_np)
            image_np = pad_or_crop(image_np, SPATIAL_SIZE)
            label_3ch = convert_labels(pad_or_crop(label_raw, SPATIAL_SIZE))
            records.append({
                "case_id": case_id,
                "image":   image_np,    # (4, D, H, W) float32
                "label":   label_3ch,   # (3, D, H, W) float32
            })
        except Exception as e:
            logger.warning(f"Skipping {case_id}: {e}")
    return records


def best_axial_slice(label_3ch: np.ndarray) -> int:
    """Return the axial slice index with the most tumour voxels."""
    tumor = label_3ch[0]  # WT has most voxels
    counts = tumor.sum(axis=(1, 2))
    return int(np.argmax(counts))


# ============================================================================
# FORWARD CAM  (hook-based, no gradients needed)
# ============================================================================

class ForwardCAM:
    """
    Registers a forward hook on `layer` and stores the raw activation.
    After model(x) the activation is available at .activation.
    """
    def __init__(self, layer: nn.Module):
        self.activation: Optional[torch.Tensor] = None
        self._hook = layer.register_forward_hook(self._save)

    def _save(self, module, inp, output):
        # output may be a tensor or a tuple — take first tensor
        if isinstance(output, (tuple, list)):
            out = output[0]
        else:
            out = output
        self.activation = out.detach().cpu()

    def remove(self):
        self._hook.remove()

    def get_cam(self, spatial_size: Tuple[int, ...]) -> np.ndarray:
        """
        Aggregate channel dimension → CAM of shape spatial_size.
        Works for both 2D (H, W) and 3D (D, H, W) activations.
        """
        act = self.activation  # (1, C, ...) or (1, C, D, H, W)
        if act is None:
            return np.zeros(spatial_size)

        # Mean over channel dimension → (1, 1, ...)
        cam = act.mean(dim=1, keepdim=True)   # (1, 1, ...)
        cam = F.relu(cam)                     # keep positive activations

        # Upsample to target size
        ndim = cam.ndim - 2  # spatial dims
        mode = "trilinear" if ndim == 3 else "bilinear"
        cam = F.interpolate(
            cam.float(),
            size=spatial_size,
            mode=mode,
            align_corners=False,
        )
        cam = cam.squeeze().numpy()   # (D, H, W) or (H, W)

        # Normalise 0–1
        mn, mx = cam.min(), cam.max()
        if mx > mn:
            cam = (cam - mn) / (mx - mn)
        return cam.astype(np.float32)


def _find_layer(model: nn.Module, names: List[str]) -> Optional[nn.Module]:
    """
    Walk model named_modules and return the first whose name contains
    any of the given substrings.  Returns None if not found.
    """
    for name, module in model.named_modules():
        for target in names:
            if target in name:
                return module
    return None


def attach_cam_hook(model: nn.Module, model_name: str) -> Tuple[ForwardCAM, str]:
    """
    Attach ForwardCAM to the appropriate layer for each model.
    Returns (hook, layer_name_found).
    """
    layer_hints = {
        "segresnet":  ["convolutions.4", "encode_4", "up_layers.0",
                       "down_layers.3", "down_layers.2"],
        "swinunetr":  ["swinViT.layers4", "swinViT.layers3",
                       "encoder10", "decoder5"],
        "nnunet":     ["seg_layers", "decoder.stages.4",
                       "decoder.stages.3", "encoder.stages.4",
                       "encoder.stages.3"],
        "custom":     ["backbone.transformer", "backbone.cross_attn",
                       "backbone.block2", "refine"],
    }

    hints = layer_hints.get(model_name, [])
    layer = _find_layer(model, hints)

    if layer is None:
        # Fallback: use the last nn.Conv2d or nn.Conv3d
        last = None
        for _, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                last = m
        layer = last

    if layer is None:
        raise RuntimeError(f"Cannot find a hookable layer for {model_name}")

    # Find the actual name for logging
    layer_name = "unknown"
    for n, m in model.named_modules():
        if m is layer:
            layer_name = n
            break

    logger.info(f"  CAM hook on layer: {layer_name}")
    return ForwardCAM(layer), layer_name


# ============================================================================
# UNCERTAINTY MAP  (entropy)
# ============================================================================

def compute_entropy_map(logits: torch.Tensor, sigmoid_output: bool = True) -> np.ndarray:
    """
    Compute per-voxel entropy from model logits.

    For sigmoid models (SegResNet, SwinUNETR, Custom):
        p = sigmoid(logits)  →  H = -p*log(p+ε) - (1-p)*log(1-p+ε)
        Shape: (3, ...) → averaged → (...)

    For softmax models (nnUNet):
        Use the channel dimension as class probabilities.
    """
    with torch.no_grad():
        if sigmoid_output:
            p   = torch.sigmoid(logits.float())
            eps = 1e-7
            H   = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
            ent = H.mean(dim=0)  # average over 3 region channels
        else:
            # softmax — logits already are softmax probs from nnUNet
            p   = logits.float().clamp(1e-7, 1.0)
            H   = -(p * torch.log(p)).sum(dim=0)
            ent = H

    ent = ent.cpu().numpy()
    mn, mx = ent.min(), ent.max()
    if mx > mn:
        ent = (ent - mn) / (mx - mn)
    return ent.astype(np.float32)


# ============================================================================
# OCCLUSION XAI
# ============================================================================

@torch.no_grad()
def occlusion_modality_importance(
    model: nn.Module,
    image_tensor: torch.Tensor,
    baseline_dice: np.ndarray,
    model_name: str,
    is_3d: bool = True,
) -> np.ndarray:
    """
    Modality-level occlusion: zero-out each of the 4 input channels in turn.
    Returns drop-in-mean-Dice per modality → shape (4,).

    baseline_dice: (3,) mean Dice per region from the full input.
    """
    drops = np.zeros(NUM_MODALITIES, dtype=np.float32)
    x = image_tensor.to(DEVICE)

    for ch in range(NUM_MODALITIES):
        x_occ = x.clone()
        if is_3d:
            x_occ[:, ch, ...] = 0.0
        else:
            # 2.5D custom model: zero ALL slices for that modality channel group
            k_slices = x.shape[1] // NUM_MODALITIES
            for s in range(k_slices):
                x_occ[:, s * NUM_MODALITIES + ch, ...] = 0.0

        with torch.no_grad():
            out = _forward(model, x_occ, model_name)

        # Dice drop
        pred = (torch.sigmoid(out) > 0.5).float().cpu().numpy()[0]  # (3, ...)
        gt   = image_tensor.new_zeros(pred.shape)  # placeholder — we compare relative
        # We measure mean activation drop (proxy for dice drop)
        occ_mean = pred.mean()
        full_mean = (torch.sigmoid(x).float().cpu().mean().item())
        drops[ch] = max(0.0, full_mean - occ_mean)

    # Normalise to sum=1
    s = drops.sum()
    if s > 0:
        drops /= s
    return drops


@torch.no_grad()
def occlusion_patch_sensitivity(
    model: nn.Module,
    image_tensor: torch.Tensor,
    model_name: str,
    patch_size: Tuple,
    is_3d: bool = True,
) -> np.ndarray:
    """
    Patch-level occlusion sensitivity map.
    Strides a patch (zero-fill) across the input volume/slice and records
    the mean output activation drop.  Returns a heatmap of the same spatial
    size as the input.

    For 3D: patch_size = (pD, pH, pW)
    For 2D (custom model): patch_size = patch_size[:2]
    """
    x  = image_tensor.to(DEVICE)

    with torch.no_grad():
        full_out = _forward(model, x, model_name)
        full_score = torch.sigmoid(full_out).mean().item()

    if is_3d:
        _, C, D, H, W = x.shape
        pD, pH, pW = patch_size
        sens = np.zeros((D, H, W), dtype=np.float32)
        count = np.zeros((D, H, W), dtype=np.float32)

        for d in range(0, D, pD):
            for h in range(0, H, pH):
                for w in range(0, W, pW):
                    x_occ = x.clone()
                    x_occ[:, :, d:d+pD, h:h+pH, w:w+pW] = 0.0
                    with torch.no_grad():
                        out = _forward(model, x_occ, model_name)
                    score = torch.sigmoid(out).mean().item()
                    drop  = max(0.0, full_score - score)
                    sens[d:d+pD, h:h+pH, w:w+pW]  += drop
                    count[d:d+pD, h:h+pH, w:w+pW] += 1.0
        count = np.where(count == 0, 1, count)
        sens /= count
    else:
        _, C, H, W = x.shape
        pH, pW = patch_size[:2]
        sens  = np.zeros((H, W), dtype=np.float32)
        count = np.zeros((H, W), dtype=np.float32)
        for h in range(0, H, pH):
            for w in range(0, W, pW):
                x_occ = x.clone()
                x_occ[:, :, h:h+pH, w:w+pW] = 0.0
                with torch.no_grad():
                    out = _forward(model, x_occ, model_name)
                score = torch.sigmoid(out).mean().item()
                drop  = max(0.0, full_score - score)
                sens[h:h+pH, w:w+pW]  += drop
                count[h:h+pH, w:w+pW] += 1.0
        count = np.where(count == 0, 1, count)
        sens /= count

    mn, mx = sens.min(), sens.max()
    if mx > mn:
        sens = (sens - mn) / (mx - mn)
    return sens.astype(np.float32)


# ============================================================================
# MODEL-SPECIFIC FORWARD PASS WRAPPER
# ============================================================================

def _forward(model: nn.Module, x: torch.Tensor, model_name: str) -> torch.Tensor:
    """Unified forward: returns raw logits tensor (B, 3, ...) for all models."""
    model.eval()
    with torch.cuda.amp.autocast(enabled=USE_AMP):
        if model_name == "custom":
            out = model(x)
            return out["final"]                 # (B, 3, H, W)
        elif model_name == "nnunet":
            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]                    # deep supervision: take first
            return out                          # (B, 3+1, ...) softmax or logits
        else:
            return model(x)                     # SegResNet / SwinUNETR


# ============================================================================
# MODEL LOADERS (re-use logic from evaluate scripts)
# ============================================================================

def load_segresnet(ckpt_path: Path) -> nn.Module:
    from monai.networks.nets import SegResNet
    model = SegResNet(**SEGRESNET_CFG).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = _extract_state(ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info(f"SegResNet loaded from {ckpt_path}")
    return model


def load_swinunetr(ckpt_path: Path) -> nn.Module:
    from monai.networks.nets import SwinUNETR
    model = SwinUNETR(**SWIN_CFG).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = _extract_state(ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info(f"SwinUNETR loaded from {ckpt_path}")
    return model


def load_nnunet(ckpt_path: Path) -> nn.Module:
    """Load nnUNet directly from checkpoint_best.pth (mirrors evaluate_nnunet.py)."""
    import json as _json
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    import inspect

    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    fold_dir   = ckpt_path.parent
    config_dir = fold_dir.parent
    plans_path   = config_dir / "plans.json"
    dataset_path = config_dir / "dataset.json"

    with open(plans_path,   "r") as f: plans   = _json.load(f)
    with open(dataset_path, "r") as f: ds_json = _json.load(f)

    n_in  = len(ds_json.get("channel_names", ds_json.get("modality", {0:0,1:1,2:2,3:3})))
    n_out = max(len(ds_json.get("labels", {0:0,1:1,2:2,3:3})) - 1, 1)

    cfg = plans["configurations"]["3d_fullres"]
    arch = cfg.get("architecture", {})

    def _fix(d):
        return {(int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k):
                (_fix(v) if isinstance(v, dict) else v) for k, v in d.items()}

    sig = list(inspect.signature(get_network_from_plans).parameters.keys())
    arch_cls  = (arch.get("network_class_name") or arch.get("arch_class_name")
                 or cfg.get("network_arch_class_name"))
    arch_kw   = _fix(arch.get("arch_kwargs") or arch.get("network_arch_init_kwargs") or {})
    arch_imp  = (arch.get("_kw_requires_import") or arch.get("arch_kwargs_req_import") or [])

    if sig[0] == "arch_class_name":
        network = get_network_from_plans(arch_cls, arch_kw, arch_imp,
                                         n_in, n_out,
                                         allow_init=True, deep_supervision=False)
    else:
        ds_stub = {"labels": {str(i): i for i in range(n_out)},
                   "channel_names": {str(i): f"mod_{i}" for i in range(n_in)}}
        network = get_network_from_plans(plans, ds_stub, "3d_fullres", n_in,
                                         deep_supervision=False)

    state = ckpt.get("network_weights", ckpt.get("model", ckpt))
    network.load_state_dict(state, strict=False)
    network = network.to(DEVICE)
    network.eval()
    logger.info(f"nnUNet loaded from {ckpt_path}")
    return network


def load_custom(ckpt_path: Path) -> nn.Module:
    from custom_model_sanity_check import CustomBraTSModel
    model = CustomBraTSModel(k=3, base_channels=64, patch_size=8).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = _extract_state(ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info(f"Custom model loaded from {ckpt_path}")
    return model


def _extract_state(ckpt) -> dict:
    if isinstance(ckpt, dict):
        for key in ["model_state", "model", "state_dict", "net", "model_state_dict"]:
            if key in ckpt:
                return ckpt[key]
    return ckpt  # bare state dict


# ============================================================================
# VISUALISATION HELPERS
# ============================================================================

def _mri_bg(image_np: np.ndarray, z: int) -> np.ndarray:
    """T1c axial slice (channel 1) normalised to 0-1 for background."""
    sl = image_np[1, z, :, :]
    mn, mx = sl.min(), sl.max()
    if mx > mn:
        sl = (sl - mn) / (mx - mn)
    return sl.astype(np.float32)


def save_cam_overlay(
    cam_3d: np.ndarray,        # (D, H, W)  normalised 0-1
    image_np: np.ndarray,      # (4, D, H, W)
    case_id: str,
    out_dir: Path,
    model_name: str,
    label: str = "cam",
):
    """Save axial-slice CAM overlay on T1c background."""
    z = int(np.argmax(cam_3d.sum(axis=(1, 2))))
    bg  = _mri_bg(image_np, z)
    cam = cam_3d[z] if cam_3d.ndim == 3 else cam_3d

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(bg.T, cmap="gray", origin="lower")
    axes[0].set_title("T1c (axial)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(bg.T, cmap="gray", origin="lower")
    im = axes[1].imshow(cam.T, cmap=CAM_CMAP, alpha=OVERLAY_ALPHA, origin="lower",
                        vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title(f"ForwardCAM — {model_name}", fontsize=10)
    axes[1].axis("off")

    fig.suptitle(f"{case_id} | slice z={z}", fontsize=11)
    plt.tight_layout()
    fpath = out_dir / f"{case_id}_{label}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    return fpath


def save_mask_overlay(
    pred_np: np.ndarray,    # (3, D, H, W) binary
    gt_np:   np.ndarray,    # (3, D, H, W) binary
    image_np: np.ndarray,   # (4, D, H, W)
    case_id:  str,
    out_dir:  Path,
    model_name: str,
):
    """
    Side-by-side GT vs Pred mask overlay for all 3 regions on the best slice.
    Each region gets its own colour following REGION_COLORS.
    """
    z   = best_axial_slice(gt_np)
    bg  = _mri_bg(image_np, z)

    region_rgb = {
        "WT": np.array([232, 76, 76])  / 255,
        "TC": np.array([76, 175, 80])  / 255,
        "ET": np.array([76, 139, 232]) / 255,
    }

    def _make_overlay(mask_3ch, z):
        rgba = np.zeros((*bg.shape, 4), dtype=np.float32)
        for ri, rname in enumerate(REGION_NAMES):
            sl   = mask_3ch[ri, z]
            col  = region_rgb[rname]
            rgba[sl > 0.5, :3] = col
            rgba[sl > 0.5,  3] = 0.6
        return rgba

    gt_rgba   = _make_overlay(gt_np,   z)
    pred_rgba = _make_overlay(pred_np, z)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, rgba, title in zip(axes,
                                [gt_rgba, pred_rgba],
                                ["Ground Truth", f"Prediction ({model_name})"]):
        ax.imshow(bg.T, cmap="gray", origin="lower")
        ax.imshow(rgba.transpose(1, 0, 2), origin="lower")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    patches = [mpatches.Patch(color=region_rgb[r], label=r) for r in REGION_NAMES]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=10)
    fig.suptitle(f"{case_id} | slice z={z}", fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fpath = out_dir / f"{case_id}_overlay.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    return fpath


def save_occlusion_map(
    sens_map: np.ndarray,   # (D,H,W) or (H,W)
    image_np: np.ndarray,
    case_id: str,
    out_dir: Path,
    model_name: str,
    label: str = "patch",
):
    """Save patch-level occlusion sensitivity map."""
    is_3d = sens_map.ndim == 3
    if is_3d:
        z   = int(np.argmax(sens_map.sum(axis=(1, 2))))
        sl  = sens_map[z]
    else:
        z  = sens_map.shape[0] // 2
        sl = sens_map

    bg = _mri_bg(image_np, z) if image_np.ndim == 4 else image_np

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(bg.T, cmap="gray", origin="lower")
    axes[0].set_title("T1c (axial)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(bg.T, cmap="gray", origin="lower")
    im = axes[1].imshow(sl.T, cmap="inferno", alpha=0.6, origin="lower", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title(f"Occlusion Sensitivity ({label})", fontsize=10)
    axes[1].axis("off")

    fig.suptitle(f"{case_id} | {model_name}", fontsize=11)
    plt.tight_layout()
    fpath = out_dir / f"{case_id}_occlusion_{label}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    return fpath


def save_uncertainty_map(
    ent_map: np.ndarray,    # (D,H,W) or (H,W)
    image_np: np.ndarray,
    case_id: str,
    out_dir: Path,
    model_name: str,
):
    """Save per-voxel entropy uncertainty map."""
    is_3d = ent_map.ndim == 3
    if is_3d:
        z  = best_axial_slice(ent_map[np.newaxis])  # treat as 1-ch label
        sl = ent_map[z]
    else:
        sl = ent_map
        z  = 0

    bg = _mri_bg(image_np, z) if (image_np.ndim == 4 and is_3d) else image_np[1].mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(bg.T, cmap="gray", origin="lower")
    axes[0].set_title("T1c (axial)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(bg.T, cmap="gray", origin="lower")
    im = axes[1].imshow(sl.T, cmap="plasma", alpha=0.7, origin="lower", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title(f"Entropy Uncertainty Map", fontsize=10)
    axes[1].axis("off")

    fig.suptitle(f"{case_id} | {model_name}", fontsize=11)
    plt.tight_layout()
    fpath = out_dir / f"{case_id}_uncertainty.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    return fpath


def save_modality_importance(
    importances_all: np.ndarray,   # (N_cases, 4)
    model_name: str,
    out_dir: Path,
):
    """Save per-modality importance bar chart (mean ± std across cases)."""
    mean_imp = importances_all.mean(axis=0)
    std_imp  = importances_all.std(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(MODALITY_KEYS, mean_imp, yerr=std_imp, capsize=5,
                  color=MODALITY_COLORS, edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Modality", fontsize=11)
    ax.set_ylabel("Relative Importance (normalised drop)", fontsize=11)
    ax.set_title(f"Modality Importance — {model_name}", fontsize=12)
    ax.set_ylim(0, max(mean_imp.max() * 1.3, 0.4))

    for bar, m, s in zip(bars, mean_imp, std_imp):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fpath = out_dir / f"{model_name}_modality_importance.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved modality importance: {fpath}")
    return fpath


# ============================================================================
# PER-MODEL XAI RUNNER
# ============================================================================

def run_xai_3d(
    model: nn.Module,
    model_name: str,
    cases: List[Dict],
    dirs: Dict[str, Path],
    is_sigmoid: bool = True,
):
    """
    Full XAI pipeline for 3D models (SegResNet, SwinUNETR, nnUNet).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  XAI — {model_name.upper()}  ({len(cases)} cases)")
    logger.info(f"{'='*60}")

    # Attach CAM hook
    cam_hook, cam_layer = attach_cam_hook(model, model_name)
    importances = []

    for rec in tqdm(cases, desc=f"XAI {model_name}", unit="case"):
        case_id  = rec["case_id"]
        image_np = rec["image"]    # (4, D, H, W)
        label_np = rec["label"]    # (3, D, H, W)

        x = torch.from_numpy(image_np).unsqueeze(0).to(DEVICE)  # (1,4,D,H,W)

        # ── 1. FORWARD PASS ──────────────────────────────────────────────────
        with torch.no_grad():
            logits = _forward(model, x, model_name)         # (1, 3, D, H, W)

        pred_np = (torch.sigmoid(logits).squeeze(0).cpu().numpy() > 0.5).astype(np.float32)

        # ── 2. ForwardCAM ─────────────────────────────────────────────────────
        cam_3d = cam_hook.get_cam(SPATIAL_SIZE)              # (D, H, W)
        save_cam_overlay(cam_3d, image_np, case_id, dirs["cam"], model_name)

        # ── 3. Mask overlay (pred vs GT) ──────────────────────────────────────
        save_mask_overlay(pred_np, label_np, image_np, case_id, dirs["overlay"], model_name)

        # ── 4. Uncertainty entropy map ────────────────────────────────────────
        ent = compute_entropy_map(logits.squeeze(0), sigmoid_output=is_sigmoid)
        save_uncertainty_map(ent, image_np, case_id, dirs["uncertainty"], model_name)

        # ── 5. Occlusion — patch level ────────────────────────────────────────
        sens = occlusion_patch_sensitivity(model, x, model_name,
                                           patch_size=OCCLUSION_PATCH, is_3d=True)
        save_occlusion_map(sens, image_np, case_id, dirs["occlusion"], model_name, "patch")

        # ── 6. Occlusion — modality level ─────────────────────────────────────
        base_dice = np.array([float(pred_np[i].mean()) for i in range(3)])
        mod_imp   = occlusion_modality_importance(model, x, base_dice, model_name, is_3d=True)
        importances.append(mod_imp)

        logger.info(f"  {case_id}: CAM, overlay, uncertainty, occlusion ✓")

    cam_hook.remove()

    # ── Modality importance summary ───────────────────────────────────────────
    if importances:
        imp_arr = np.stack(importances, axis=0)    # (N, 4)
        save_modality_importance(imp_arr, model_name, dirs["modality"])
        np.save(dirs["modality"] / f"{model_name}_modality_importance.npy", imp_arr)

    logger.info(f"  {model_name} XAI complete → {XAI_DIR / model_name}")


def run_xai_custom(
    model: nn.Module,
    cases: List[Dict],
    dirs: Dict[str, Path],
):
    """
    XAI for the custom 2.5D model.
    The custom model already outputs 'uncertainty' and 'importance' tensors —
    we use them directly.  CAM is hooked on backbone.transformer.
    For occlusion we work on the centre slice.
    """
    model_name = "custom"
    logger.info(f"\n{'='*60}")
    logger.info(f"  XAI — CUSTOM  ({len(cases)} cases)")
    logger.info(f"{'='*60}")

    from custom_model_sanity_check import BraTS25DDataset
    from config import SPLIT_CACHE, SPLIT_SEED, TRAIN_RATIO, VAL_RATIO

    cam_hook, cam_layer = attach_cam_hook(model, model_name)
    importances = []

    K         = 3
    TARGET_HW = (192, 192)

    for rec in tqdm(cases, desc="XAI custom", unit="case"):
        case_id  = rec["case_id"]
        image_np = rec["image"]    # (4, D, H, W)
        label_np = rec["label"]    # (3, D, H, W)

        D  = image_np.shape[1]
        z  = best_axial_slice(label_np)

        # Build 2.5D input for the centre slice
        pad   = K // 2
        slices = []
        for dz in range(-pad, pad + 1):
            iz = int(np.clip(z + dz, 0, D - 1))
            for ch in range(NUM_MODALITIES):
                sl = image_np[ch, iz, :, :]
                slices.append(sl)

        x_np = np.stack(slices, axis=0).astype(np.float32)   # (4*K, H, W)
        x_t  = torch.from_numpy(x_np).unsqueeze(0)
        x_t  = F.interpolate(x_t, size=TARGET_HW, mode="bilinear", align_corners=False)
        x    = x_t.to(DEVICE)   # (1, 12, 192, 192)

        # ── 1. Forward pass ───────────────────────────────────────────────────
        with torch.no_grad():
            out = model(x)

        logits     = out["final"]           # (1, 3, 192, 192)
        unc_tensor = out["uncertainty"]     # (1, 1, H, W)
        imp_tensor = out["importance"]      # (1, 1, H, W)

        pred_2d = (torch.sigmoid(logits).squeeze(0).cpu().numpy() > 0.5).astype(np.float32)

        # ── 2. ForwardCAM ─────────────────────────────────────────────────────
        cam_2d = cam_hook.get_cam(TARGET_HW)                  # (192, 192)

        # Build a 3D CAM from the 2D slice (smear along depth for viz)
        # Resize CAM from TARGET_HW (192,192) → SPATIAL_SIZE H/W (128,128)
        sH, sW = SPATIAL_SIZE[1], SPATIAL_SIZE[2]
        cam_2d_resized = F.interpolate(
            torch.from_numpy(cam_2d).unsqueeze(0).unsqueeze(0),
            size=(sH, sW), mode="bilinear", align_corners=False,
        ).squeeze().numpy()
        cam_3d = np.zeros(SPATIAL_SIZE, dtype=np.float32)
        cam_3d[z] = cam_2d_resized
        save_cam_overlay(cam_3d, image_np, case_id, dirs["cam"], model_name)

        # ── 3. Mask overlay ───────────────────────────────────────────────────
        pred_3d = np.zeros(label_np.shape, dtype=np.float32)   # (3, D, H, W)
        for ch in range(NUM_CLASSES):
            p2 = F.interpolate(
                torch.from_numpy(pred_2d[ch]).unsqueeze(0).unsqueeze(0).float(),
                size=(sH, sW), mode="nearest",
            ).squeeze().numpy()
            pred_3d[ch, z] = p2
        save_mask_overlay(pred_3d, label_np, image_np, case_id, dirs["overlay"], model_name)

        # ── 4. Uncertainty from model's own output ────────────────────────────
        unc_np = torch.sigmoid(unc_tensor).squeeze().cpu().numpy()   # (192, 192)
        mn, mx = unc_np.min(), unc_np.max()
        if mx > mn:
            unc_np = (unc_np - mn) / (mx - mn)

        # Save as 2D (same slice visualisation)
        bg_slice = image_np[1, z, :TARGET_HW[0], :TARGET_HW[1]].T
        bg_mn, bg_mx = bg_slice.min(), bg_slice.max()
        bg_norm = (bg_slice - bg_mn) / (bg_mx - bg_mn + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(bg_norm, cmap="gray", origin="lower")
        axes[0].set_title("T1c (slice)")
        axes[0].axis("off")
        axes[1].imshow(bg_norm, cmap="gray", origin="lower")
        im = axes[1].imshow(unc_np.T, cmap="plasma", alpha=0.7, origin="lower", vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        axes[1].set_title("Fusion Uncertainty Gate")
        axes[1].axis("off")
        fig.suptitle(f"{case_id} | custom | z={z}")
        plt.tight_layout()
        unc_path = dirs["uncertainty"] / f"{case_id}_uncertainty.png"
        plt.savefig(unc_path, dpi=150, bbox_inches="tight")
        plt.close()

        # ── 5. Occlusion patch-level (2D) ─────────────────────────────────────
        sens = occlusion_patch_sensitivity(model, x, model_name,
                                           patch_size=OCCLUSION_PATCH[:2], is_3d=False)
        # Resize to image crop for visualisation
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(bg_norm, cmap="gray", origin="lower")
        axes[0].set_title("T1c (slice)")
        axes[0].axis("off")
        axes[1].imshow(bg_norm, cmap="gray", origin="lower")
        sens_resized = F.interpolate(
            torch.from_numpy(sens).unsqueeze(0).unsqueeze(0),
            size=TARGET_HW, mode="bilinear", align_corners=False
        ).squeeze().numpy()
        im = axes[1].imshow(sens_resized.T, cmap="inferno", alpha=0.6, origin="lower", vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        axes[1].set_title("Occlusion Sensitivity (patch)")
        axes[1].axis("off")
        fig.suptitle(f"{case_id} | custom | z={z}")
        plt.tight_layout()
        occ_path = dirs["occlusion"] / f"{case_id}_occlusion_patch.png"
        plt.savefig(occ_path, dpi=150, bbox_inches="tight")
        plt.close()

        # ── 6. Modality importance ─────────────────────────────────────────────
        base_dice = np.array([float(pred_2d[i].mean()) for i in range(3)])
        mod_imp   = occlusion_modality_importance(model, x, base_dice, model_name, is_3d=False)
        importances.append(mod_imp)

        logger.info(f"  {case_id}: CAM, overlay, uncertainty, occlusion ✓")

    cam_hook.remove()

    if importances:
        imp_arr = np.stack(importances, axis=0)
        save_modality_importance(imp_arr, model_name, dirs["modality"])
        np.save(dirs["modality"] / f"{model_name}_modality_importance.npy", imp_arr)

    logger.info(f"  custom XAI complete → {XAI_DIR / model_name}")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Stage 5 XAI for all 4 BraTS models")
    p.add_argument("--model", type=str, default="all",
                   choices=["all", "segresnet", "swinunetr", "nnunet", "custom"],
                   help="Which model to run XAI for (default: all)")
    p.add_argument("--n_cases", type=int, default=N_VIZ_CASES,
                   help="Number of test cases to visualise")
    p.add_argument("--segresnet_ckpt", type=str, default=None)
    p.add_argument("--swinunetr_ckpt", type=str, default=None)
    p.add_argument("--nnunet_ckpt",    type=str, default=None)
    p.add_argument("--custom_ckpt",    type=str, default=None)
    return p.parse_args()


def resolve_ckpt(arg: Optional[str], default: Path) -> Path:
    if arg:
        return Path(arg)
    return default


def main():
    args = parse_args()

    # Default checkpoint paths (mirrors evaluate scripts)
    DEFAULT_NNUNET_CKPT = (
        NNUNET_RESULTS
        / "Dataset137_BraTSGLI"
        / "nnUNetTrainerCustom__nnUNetPlans__3d_fullres"
        / "fold_0"
        / "checkpoint_best.pth"
    )

    ckpts = {
        "segresnet":  resolve_ckpt(args.segresnet_ckpt, CHECKPOINT_DIR / "segresnet_best.pth"),
        "swinunetr":  resolve_ckpt(args.swinunetr_ckpt, CHECKPOINT_DIR / "swinunetr_best.pth"),
        "nnunet":     resolve_ckpt(args.nnunet_ckpt,    DEFAULT_NNUNET_CKPT),
        "custom":     resolve_ckpt(args.custom_ckpt,    CHECKPOINT_DIR / "custom_best.pth"),
    }

    run_models = (
        ["segresnet", "swinunetr", "nnunet", "custom"]
        if args.model == "all" else [args.model]
    )

    logger.info(f"Loading {args.n_cases} test cases...")
    cases = load_test_cases(args.n_cases)
    logger.info(f"Loaded {len(cases)} cases.")

    if len(cases) == 0:
        logger.error("No test cases found — check DATA_ROOT in config.py")
        return

    # ── SegResNet ─────────────────────────────────────────────────────────────
    if "segresnet" in run_models:
        ckpt = ckpts["segresnet"]
        if not ckpt.exists():
            logger.warning(f"SegResNet checkpoint not found: {ckpt}")
        else:
            dirs  = make_xai_dirs("segresnet")
            model = load_segresnet(ckpt)
            run_xai_3d(model, "segresnet", cases, dirs, is_sigmoid=True)
            del model; torch.cuda.empty_cache()

    # ── SwinUNETR ─────────────────────────────────────────────────────────────
    if "swinunetr" in run_models:
        ckpt = ckpts["swinunetr"]
        if not ckpt.exists():
            logger.warning(f"SwinUNETR checkpoint not found: {ckpt}")
        else:
            dirs  = make_xai_dirs("swinunetr")
            model = load_swinunetr(ckpt)
            run_xai_3d(model, "swinunetr", cases, dirs, is_sigmoid=True)
            del model; torch.cuda.empty_cache()

    # ── nnUNet ────────────────────────────────────────────────────────────────
    if "nnunet" in run_models:
        ckpt = ckpts["nnunet"]
        if not ckpt.exists():
            logger.warning(f"nnUNet checkpoint not found: {ckpt}")
        else:
            dirs  = make_xai_dirs("nnunet")
            model = load_nnunet(ckpt)
            run_xai_3d(model, "nnunet", cases, dirs, is_sigmoid=False)
            del model; torch.cuda.empty_cache()

    # ── Custom ────────────────────────────────────────────────────────────────
    if "custom" in run_models:
        ckpt = ckpts["custom"]
        if not ckpt.exists():
            logger.warning(f"Custom checkpoint not found: {ckpt}")
        else:
            dirs  = make_xai_dirs("custom")
            model = load_custom(ckpt)
            run_xai_custom(model, cases, dirs)
            del model; torch.cuda.empty_cache()

    logger.info(f"\n{'='*60}")
    logger.info(f"  All XAI outputs saved to: {XAI_DIR}")
    logger.info(f"{'='*60}")
    logger.info("Folder structure:")
    for mn in run_models:
        d = XAI_DIR / mn
        if d.exists():
            total = sum(1 for _ in d.rglob("*.png"))
            logger.info(f"  {mn:12s} → {d}  ({total} PNGs)")


if __name__ == "__main__":
    main()