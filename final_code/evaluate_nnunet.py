"""
evaluate_nnunet.py — nnUNet v2 evaluation via direct checkpoint loading.

Loads nnUNet's trained network directly from checkpoint_best.pth without
needing the subprocess inference pipeline. Uses nnUNet's own model
reconstruction utilities so the architecture is always correct.

Outputs saved to config.RESULTS_DIR / "nnunet":
  metrics_per_case.csv         — Dice, IoU, HD95 per case per region
  metrics_summary.csv          — mean ± std per region
  classification_metrics.csv   — precision, recall, F1, accuracy per region
  metrics_summary.json         — full summary in JSON
  roc_curves/                  — ROC-AUC PNG per region
  dice_boxplot.png             — Dice distribution per region
  dice_bar.png                 — mean Dice bar chart with error bars
  dice_per_case.png            — per-case Dice sorted by mean
  pred_masks/                  — predicted NIfTI masks for all test cases
  overlay_viz/                 — axial slice overlays GT vs pred (best+worst)

Usage:
    python evaluate_nnunet.py --checkpoint "C:\\path\\to\\checkpoint_best.pth"
    python evaluate_nnunet.py --checkpoint "C:\\path\\to\\checkpoint_best.pth" --n_viz 3
"""

import argparse
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

from config import (
    DATA_ROOT, DEVICE, N_VIZ_CASES,
    NNUNET_PREPROCESSED, NNUNET_RAW, NNUNET_RESULTS,
    REGION_NAMES, RESULTS_DIR,
)
from dataset import (
    build_splits, convert_labels, discover_cases,
    get_case_paths_for_split,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

os.environ["nnUNet_raw"]          = str(NNUNET_RAW)
os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
os.environ["nnUNet_results"]      = str(NNUNET_RESULTS)

# ---------------------------------------------------------------------------
# Output dirs
# ---------------------------------------------------------------------------
OUT_DIR     = RESULTS_DIR / "nnunet"
ROC_DIR     = OUT_DIR / "roc_curves"
MASK_DIR    = OUT_DIR / "pred_masks"
OVERLAY_DIR = OUT_DIR / "overlay_viz"
for _d in [ROC_DIR, MASK_DIR, OVERLAY_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

REGION_COLORS = {"WT": "#E84C4C", "TC": "#4CAF50", "ET": "#4C8BE8"}

DEFAULT_CKPT = (
    NNUNET_RESULTS
    / "Dataset137_BraTSGLI"
    / "nnUNetTrainerCustom__nnUNetPlans__3d_fullres"
    / "fold_0"
    / "checkpoint_best.pth"
)


# ===========================================================================
# 1. MODEL LOADING — direct from nnUNet checkpoint
# ===========================================================================

def _get_num_input_channels(dataset_json: dict) -> int:
    for key in ("channel_names", "modality"):
        val = dataset_json.get(key)
        if val is not None:
            return len(val)
    logger.warning("Could not determine input channels from dataset.json — defaulting to 4")
    return 4


def _get_num_output_channels(dataset_json: dict) -> int:
    """
    Returns foreground output channels only (background excluded).
    BraTS: 4 labels (bg+3) -> 3 output channels, which matches the checkpoint.
    """
    labels = dataset_json.get("labels")
    if labels is not None:
        n_labels = len(labels)
        n_out = max(n_labels - 1, 1)  # subtract background
        logger.info(f"  Output channels: {n_labels} labels - 1 background = {n_out}")
        return n_out
    logger.warning("labels missing from dataset.json -- defaulting to 3")
    return 3


def _fix_arch_kwargs(kwargs: dict) -> dict:
    """
    nnUNet stores some arch_kwargs with integer-string keys (e.g. "0", "1")
    because JSON only allows string keys. Network constructors expect int keys.
    Converts recursively.
    """
    fixed = {}
    for k, v in kwargs.items():
        new_k = int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k
        if isinstance(v, dict):
            v = _fix_arch_kwargs(v)
        fixed[new_k] = v
    return fixed


def _build_network(
    plans: dict,
    num_input_channels: int,
    num_output_channels: int,
    configuration: str = "3d_fullres",
) -> torch.nn.Module:
    """
    Build the nnUNet network from plans.json.

    The installed version has this signature (confirmed from traceback):
        get_network_from_plans(
            arch_class_name,        # str  — e.g. "PlainConvUNet"
            arch_kwargs,            # dict — network init kwargs from plans
            arch_kwargs_req_import, # list — modules to import for kwargs
            input_channels,         # int
            output_channels,        # int
            allow_init=True,        # bool
            deep_supervision=True,  # bool
        )

    The KeyError on '3' was caused by integer-string keys in arch_kwargs
    (JSON stores dict keys as strings; the UNet expects int keys for its
    kernel_sizes / strides dicts). _fix_arch_kwargs() handles this.
    """
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    import inspect

    sig_params = list(inspect.signature(get_network_from_plans).parameters.keys())
    logger.info(f"  get_network_from_plans params: {sig_params}")

    cfg = plans["configurations"][configuration]

    # ------------------------------------------------------------------
    # Variant A — newest API: explicit arch fields
    #   params start with arch_class_name
    # ------------------------------------------------------------------
    if sig_params[0] == "arch_class_name":
        # nnUNet stores architecture spec in cfg["architecture"] as:
        #   {"arch_class_name": "...", "arch_kwargs": {...},
        #    "arch_kwargs_req_import": [...]}
        # Older versions stored them flat in cfg. We check both.
        arch_block = cfg.get("architecture", {})

        # Try every key variant seen across nnUNet v2 releases
        arch_class_name = (
            arch_block.get("network_class_name")           # current nnUNet v2
            or arch_block.get("arch_class_name")
            or arch_block.get("network_arch_class_name")
            or cfg.get("network_arch_class_name")
            or cfg.get("arch_class_name")
        )
        arch_kwargs = (
            arch_block.get("arch_kwargs")                  # current nnUNet v2
            or arch_block.get("network_arch_init_kwargs")
            or cfg.get("network_arch_init_kwargs")
            or cfg.get("arch_kwargs", {})
        )
        arch_kwargs_req_import = (
            arch_block.get("_kw_requires_import")          # current nnUNet v2
            or arch_block.get("arch_kwargs_req_import")
            or arch_block.get("network_arch_init_kwargs_req_import")
            or cfg.get("network_arch_init_kwargs_req_import")
            or cfg.get("arch_kwargs_req_import", [])
        )

        if arch_class_name is None:
            raise KeyError(
                f"Cannot find arch class name in plans['{configuration}']['architecture'].\n"
                f"cfg['architecture'] keys: {list(arch_block.keys())}\n"
                f"cfg keys: {list(cfg.keys())}"
            )

        # FIX: convert string integer keys → real int keys so the UNet
        # constructor can index them with integers (e.g. kernel_sizes[0])
        arch_kwargs = _fix_arch_kwargs(arch_kwargs)

        logger.info(f"  Arch class : {arch_class_name}")
        logger.info(f"  Arch kwargs keys: {list(arch_kwargs.keys())}")

        return get_network_from_plans(
            arch_class_name,
            arch_kwargs,
            arch_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=False,
        )

    # ------------------------------------------------------------------
    # Variant B — plans-object style with output_channels
    # ------------------------------------------------------------------
    if "output_channels" in sig_params or "num_classes" in sig_params:
        dataset_stub = {
            "labels":        {str(i): i for i in range(num_output_channels)},
            "channel_names": {str(i): f"mod_{i}" for i in range(num_input_channels)},
        }
        try:
            return get_network_from_plans(
                plans, dataset_stub, configuration,
                num_input_channels, num_output_channels,
                deep_supervision=False,
            )
        except TypeError:
            pass

    # ------------------------------------------------------------------
    # Variant C — plans-object style without output_channels
    # ------------------------------------------------------------------
    dataset_stub = {
        "labels":        {str(i): i for i in range(num_output_channels)},
        "channel_names": {str(i): f"mod_{i}" for i in range(num_input_channels)},
    }
    return get_network_from_plans(
        plans, dataset_stub, configuration,
        num_input_channels,
        deep_supervision=False,
    )


def load_nnunet_model(ckpt_path: Path) -> Tuple[torch.nn.Module, dict, dict]:
    """
    Reconstruct the nnUNet network from checkpoint_best.pth and load weights.
    """
    logger.info(f"Loading nnUNet checkpoint: {ckpt_path}")

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    logger.info(f"  Checkpoint keys: {list(ckpt.keys())}")
    logger.info(f"  Epoch: {ckpt.get('current_epoch', '?')}")

    fold_dir   = ckpt_path.parent
    config_dir = fold_dir.parent

    plans_path   = config_dir / "plans.json"
    dataset_path = config_dir / "dataset.json"

    if not plans_path.exists():
        raise FileNotFoundError(f"plans.json not found at {plans_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset.json not found at {dataset_path}")

    with open(plans_path,   "r") as f:
        plans = json.load(f)
    with open(dataset_path, "r") as f:
        dataset_json = json.load(f)

    logger.info(f"  Plans:   {plans_path}")
    logger.info(f"  Dataset: {dataset_path}")

    num_input_channels  = _get_num_input_channels(dataset_json)
    num_output_channels = _get_num_output_channels(dataset_json)

    logger.info(f"  Input channels : {num_input_channels}")
    logger.info(f"  Output channels: {num_output_channels}")

    network = _build_network(plans, num_input_channels, num_output_channels)
    network = network.to(DEVICE)

    # Load weights
    state_key = "network_weights"
    if state_key not in ckpt:
        for k in ["model", "state_dict", "model_state", "network"]:
            if k in ckpt:
                state_key = k
                logger.warning(f"  'network_weights' not found, using '{k}'")
                break
        else:
            raise KeyError(
                f"Cannot find state dict in checkpoint. Keys: {list(ckpt.keys())}"
            )

    state = ckpt[state_key]
    if next(iter(state.keys())).startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        logger.info("  Stripped 'module.' prefix")

    network.load_state_dict(state, strict=True)
    network.eval()
    logger.info("  nnUNet network loaded and ready.")

    return network, ckpt, plans


# ===========================================================================
# 2. PREPROCESSING — match nnUNet's internal pipeline
# ===========================================================================

def preprocess_case_for_nnunet(
    case_dir: Path,
    plans:    dict,
    configuration: str = "3d_fullres",
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Preprocess one BraTS case to match nnUNet's training preprocessing.

    Returns:
        x      : float32 tensor (1, 4, D, H, W) ready for the network
        gt_3ch : (3, D, H, W) binary float32 ground truth
    """
    from config import MODALITY_KEYS, SEG_SUFFIX

    case_id = case_dir.name

    vols = []
    for suffix in MODALITY_KEYS:
        fpath = case_dir / f"{case_id}-{suffix}.nii.gz"
        v = nib.load(str(fpath)).get_fdata(dtype=np.float32)
        mask = v > 0
        if mask.any():
            v = (v - v[mask].mean()) / max(v[mask].std(), 1e-8)
        vols.append(v)

    image = np.stack(vols, axis=0)   # (4, H, W, D)

    seg_path = case_dir / f"{case_id}-{SEG_SUFFIX}.nii.gz"
    seg      = nib.load(str(seg_path)).get_fdata().astype(np.int8)  # (H,W,D)

    image = image.transpose(0, 3, 1, 2)   # (4, D, H, W)
    seg   = seg.transpose(2, 0, 1)         # (D, H, W)

    gt_3ch = convert_labels(seg)           # (3, D, H, W)

    nonzero = image.sum(0) != 0
    if nonzero.any():
        coords = np.argwhere(nonzero)
        d0,h0,w0 = coords.min(axis=0)
        d1,h1,w1 = coords.max(axis=0) + 1
        image  = image[:, d0:d1, h0:h1, w0:w1]
        seg    = seg[d0:d1, h0:h1, w0:w1]
        gt_3ch = gt_3ch[:, d0:d1, h0:h1, w0:w1]

    x = torch.from_numpy(image).float().unsqueeze(0).to(DEVICE)  # (1,4,D,H,W)
    return x, gt_3ch


# ===========================================================================
# 3. INFERENCE — sliding window
# ===========================================================================

@torch.no_grad()
def run_inference(
    network:  torch.nn.Module,
    case_dir: Path,
    plans:    dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run sliding-window inference using MONAI's implementation.

    Returns:
        pred_bin  : (3, D, H, W) binary float32
        pred_prob : (3, D, H, W) sigmoid probabilities float32
        gt_3ch    : (3, D, H, W) ground-truth binary float32
    """
    from monai.inferers import sliding_window_inference

    x, gt_3ch = preprocess_case_for_nnunet(case_dir, plans)

    try:
        cfg        = plans["configurations"]["3d_fullres"]
        patch_size = tuple(cfg["patch_size"])
    except (KeyError, TypeError):
        patch_size = (128, 128, 128)
        logger.debug(f"Could not read patch size from plans, using {patch_size}")

    with torch.amp.autocast("cuda", enabled=True):
        logits = sliding_window_inference(
            inputs=x,
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=network,
            overlap=0.5,
            mode="gaussian",
        )   # (1, C, D, H, W)  where C is 4 (bg + 3 regions) or 3

    logits_np = logits[0].cpu().float()   # (C, D, H, W)

    # If the model outputs 4 channels (background + 3 tumour regions),
    # drop channel 0 (background) so we get exactly 3 channels: WT, TC, ET
    if logits_np.shape[0] == 4:
        logits_np = logits_np[1:]          # (3, D, H, W)

    prob     = torch.sigmoid(logits_np).numpy()
    pred_bin = (prob > 0.5).astype(np.float32)

    # Resize pred to match gt if shapes differ (due to bounding box crop)
    if pred_bin.shape != gt_3ch.shape:
        p_t = torch.from_numpy(prob).unsqueeze(0).float()
        p_t = F.interpolate(p_t, size=gt_3ch.shape[1:],
                            mode="trilinear", align_corners=False)
        prob     = p_t.squeeze(0).numpy()
        pred_bin = (prob > 0.5).astype(np.float32)

    return pred_bin, prob, gt_3ch


# ===========================================================================
# 4. METRICS
# ===========================================================================

def dice_coeff(pred, gt, smooth=1e-5):
    p, g = pred.astype(bool), gt.astype(bool)
    return float((2*(p&g).sum()+smooth)/(p.sum()+g.sum()+smooth))

def iou_score(pred, gt, smooth=1e-5):
    p, g = pred.astype(bool), gt.astype(bool)
    return float(((p&g).sum()+smooth)/((p|g).sum()+smooth))

def hausdorff95(pred, gt):
    try:
        from scipy.ndimage import distance_transform_edt
        p, g = pred.astype(bool), gt.astype(bool)
        if not p.any() and not g.any(): return 0.0
        if not p.any() or  not g.any(): return float("nan")
        d1 = np.percentile(distance_transform_edt(~p)[g], 95)
        d2 = np.percentile(distance_transform_edt(~g)[p], 95)
        return float(max(d1, d2))
    except Exception:
        return float("nan")

def compute_all_metrics(pred_bin, gt_3ch):
    return {
        r: {"dice": dice_coeff(pred_bin[i], gt_3ch[i]),
            "iou":  iou_score(pred_bin[i],  gt_3ch[i]),
            "hd95": hausdorff95(pred_bin[i], gt_3ch[i])}
        for i, r in enumerate(REGION_NAMES)
    }


# ===========================================================================
# 5. EVALUATION LOOP
# ===========================================================================

def evaluate_all_cases(
    network:   torch.nn.Module,
    test_dirs: List[Path],
    plans:     dict,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    rows      = []
    probs_all = {r: [] for r in REGION_NAMES}
    gts_all   = {r: [] for r in REGION_NAMES}

    pbar = tqdm(test_dirs, desc="Inference + metrics", unit="case")
    for case_dir in pbar:
        case_id = case_dir.name
        pbar.set_postfix(case=case_id[-14:])
        try:
            pred_bin, prob, gt_3ch = run_inference(network, case_dir, plans)
            metrics = compute_all_metrics(pred_bin, gt_3ch)

            for region in REGION_NAMES:
                m = metrics[region]
                rows.append({"case_id": case_id, "region": region,
                             "dice": m["dice"], "iou": m["iou"], "hd95": m["hd95"]})

            for i, region in enumerate(REGION_NAMES):
                p_flat = prob[i].ravel()
                g_flat = gt_3ch[i].ravel().astype(np.uint8)
                if len(p_flat) > 50_000:
                    idx = np.random.choice(len(p_flat), 50_000, replace=False)
                    p_flat, g_flat = p_flat[idx], g_flat[idx]
                probs_all[region].append(p_flat)
                gts_all[region].append(g_flat)

            _save_nifti_mask(pred_bin, case_dir)

        except Exception as e:
            logger.warning(f"  Skipping {case_id}: {e}")

    return pd.DataFrame(rows), probs_all, gts_all


def _save_nifti_mask(pred_bin, case_dir):
    try:
        case_id = case_dir.name
        ref     = nib.load(str(case_dir / f"{case_id}-seg.nii.gz"))
        label   = np.zeros(pred_bin.shape[1:], dtype=np.uint8)
        label[pred_bin[0] > 0.5] = 1
        label[pred_bin[1] > 0.5] = 2
        label[pred_bin[2] > 0.5] = 3
        nib.save(
            nib.Nifti1Image(label.transpose(1, 2, 0), ref.affine, ref.header),
            str(MASK_DIR / f"{case_id}_nnunet_pred.nii.gz"),
        )
    except Exception as e:
        logger.debug(f"Mask save failed for {case_dir.name}: {e}")


# ===========================================================================
# 6. CLASSIFICATION METRICS
# ===========================================================================

def classification_metrics(df):
    rows = []
    for region in REGION_NAMES:
        sub    = df[df["region"] == region]["dice"].values
        y_true = (sub >  0.0).astype(int)
        y_pred = (sub >= 0.5).astype(int)
        if y_true.sum() == 0:
            continue
        rows.append({
            "region":    region, "n_cases": len(sub),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true,    y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_true,        y_pred, zero_division=0), 4),
            "accuracy":  round(accuracy_score(y_true,  y_pred),                  4),
            "mean_dice": round(float(sub.mean()), 4),
            "std_dice":  round(float(sub.std()),  4),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 7. ROC-AUC
# ===========================================================================

def compute_and_plot_roc(probs_all, gts_all):
    aucs = {}
    for region in REGION_NAMES:
        p     = np.concatenate(probs_all[region])
        g     = np.concatenate(gts_all[region]).astype(int)
        color = REGION_COLORS[region]
        if g.sum() == 0 or (1-g).sum() == 0:
            aucs[region] = float("nan"); continue
        try:
            auc         = roc_auc_score(g, p)
            fpr, tpr, _ = roc_curve(g, p)
            aucs[region] = round(float(auc), 4)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color=color, lw=2, label=f"nnUNet (AUC={auc:.4f})")
            ax.plot([0,1],[0,1],"k--",lw=0.8,label="Random")
            ax.fill_between(fpr, tpr, alpha=0.08, color=color)
            ax.set_xlabel("False positive rate", fontsize=11)
            ax.set_ylabel("True positive rate",  fontsize=11)
            ax.set_title(f"ROC curve — {region}", fontsize=13)
            ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(ROC_DIR / f"roc_{region}.png", dpi=150)
            plt.close()
            logger.info(f"  {region}: AUC = {auc:.4f}")
        except Exception as e:
            logger.warning(f"  ROC failed for {region}: {e}")
            aucs[region] = float("nan")
    return aucs


# ===========================================================================
# 8. PLOTS
# ===========================================================================

def plot_dice_boxplot(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    data   = [df[df["region"]==r]["dice"].dropna().values for r in REGION_NAMES]
    colors = [REGION_COLORS[r] for r in REGION_NAMES]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2.5),
                    whiskerprops=dict(linewidth=1.4), capprops=dict(linewidth=1.4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticklabels(REGION_NAMES, fontsize=12)
    ax.set_ylabel("Dice coefficient", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("nnUNet — Dice on test set", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    for i, (vals, region) in enumerate(zip(data, REGION_NAMES), start=1):
        if len(vals):
            ax.text(i, 1.01, f"μ={vals.mean():.3f}",
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dice_boxplot.png", dpi=150); plt.close()
    logger.info(f"Saved: {OUT_DIR / 'dice_boxplot.png'}")


def plot_dice_bar(df):
    means  = [df[df["region"]==r]["dice"].mean() for r in REGION_NAMES]
    stds   = [df[df["region"]==r]["dice"].std()  for r in REGION_NAMES]
    colors = [REGION_COLORS[r] for r in REGION_NAMES]
    x      = np.arange(len(REGION_NAMES))
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.82,
                  width=0.5, error_kw=dict(linewidth=1.5))
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x()+bar.get_width()/2, mean+std+0.015,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(REGION_NAMES, fontsize=12)
    ax.set_ylabel("Mean Dice ± std", fontsize=11); ax.set_ylim(0, 1.15)
    ax.set_title("nnUNet — Mean Dice per region", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dice_bar.png", dpi=150); plt.close()
    logger.info(f"Saved: {OUT_DIR / 'dice_bar.png'}")


def plot_per_case_dice(df):
    pivot = df.pivot_table(index="case_id", columns="region", values="dice")
    pivot["mean"] = pivot[REGION_NAMES].mean(axis=1)
    pivot = pivot.sort_values("mean")
    fig, ax = plt.subplots(figsize=(max(10, len(pivot)*0.4), 5))
    x = np.arange(len(pivot))
    for region in REGION_NAMES:
        ax.plot(x, pivot[region].values, marker="o", markersize=4,
                linewidth=1.2, label=region, color=REGION_COLORS[region])
    ax.set_xticks(x)
    ax.set_xticklabels([c[-12:] for c in pivot.index], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Dice", fontsize=11); ax.set_ylim(-0.05, 1.05)
    ax.set_title("nnUNet — Per-case Dice (sorted by mean)", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dice_per_case.png", dpi=150); plt.close()
    logger.info(f"Saved: {OUT_DIR / 'dice_per_case.png'}")


def plot_overlays(network, test_dirs, df, plans, n_viz):
    mean_dice = df.groupby("case_id")["dice"].mean().reset_index().sort_values("dice")
    worst_ids = mean_dice.head(n_viz)["case_id"].tolist()
    best_ids  = mean_dice.tail(n_viz)["case_id"].tolist()
    dir_map   = {cd.name: cd for cd in test_dirs}

    for case_id, tag in [(c,"worst") for c in worst_ids]+[(c,"best") for c in best_ids]:
        if case_id not in dir_map: continue
        case_dir = dir_map[case_id]
        try:
            pred_bin, _, gt_3ch = run_inference(network, case_dir, plans)
            z = int(gt_3ch[0].sum(axis=(1,2)).argmax())

            flair   = nib.load(str(case_dir/f"{case_id}-t2f.nii.gz")).get_fdata(dtype=np.float32)
            bg      = flair[:, :, min(z, flair.shape[2]-1)]
            bg_norm = (bg-bg.min())/(bg.max()-bg.min()+1e-8)

            fig, axes = plt.subplots(2, 3, figsize=(13, 8))
            for col, region in enumerate(REGION_NAMES):
                for row, (mask, label) in enumerate([
                    (gt_3ch[col, z],   "Ground truth"),
                    (pred_bin[col, z], "nnUNet pred"),
                ]):
                    ax = axes[row][col]
                    ax.imshow(bg_norm.T, cmap="gray", origin="lower", aspect="auto")
                    if mask.any():
                        ax.contour(mask.T, levels=[0.5],
                                   colors=[REGION_COLORS[region]], linewidths=1.8)
                    ax.set_title(f"{region} — {label}", fontsize=9)
                    ax.axis("off")

            case_dice = {r: df[(df["case_id"]==case_id)&(df["region"]==r)]["dice"].values
                         for r in REGION_NAMES}
            dice_str = "  ".join([f"{r}={v[0]:.3f}" if len(v) else f"{r}=N/A"
                                   for r,v in case_dice.items()])
            mean_d = mean_dice[mean_dice["case_id"]==case_id]["dice"].values[0]
            plt.suptitle(f"nnUNet | {case_id} | {tag.upper()} | mean={mean_d:.3f}\n{dice_str}",
                         fontsize=10)
            plt.tight_layout()
            plt.savefig(OVERLAY_DIR/f"{tag}_{case_id}.png", dpi=130, bbox_inches="tight")
            plt.close()
            logger.info(f"  Saved overlay: {tag}_{case_id}.png")
        except Exception as e:
            logger.warning(f"  Overlay failed for {case_id}: {e}")


def plot_training_curve(ckpt):
    log = ckpt.get("logging", None)
    if not log:
        logger.info("No training log in checkpoint — skipping curve.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if "train_losses" in log:
        axes[0].plot(log["train_losses"], label="train loss", color="#4C8BE8")
    if "val_losses" in log:
        axes[0].plot(log["val_losses"], label="val loss", color="#E84C4C")
    axes[0].set_title("Loss curve", fontsize=12)
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    for key in ["mean_fg_dice", "ema_fg_dice", "dice_per_class_or_region"]:
        if key in log:
            vals = log[key]
            if isinstance(vals[0], list):
                for i, region in enumerate(REGION_NAMES):
                    axes[1].plot([v[i] if i < len(v) else None for v in vals],
                                 label=f"Dice {region}", color=list(REGION_COLORS.values())[i])
            else:
                axes[1].plot(vals, label="mean fg Dice", color="#4CAF50")
            break
    axes[1].set_title("Validation Dice", fontsize=12)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0,1)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.suptitle("nnUNet — Training history", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_curve.png", dpi=150); plt.close()
    logger.info(f"Saved: {OUT_DIR / 'training_curve.png'}")


# ===========================================================================
# 9. SUMMARY PRINT
# ===========================================================================

def print_summary(df, cls_df, aucs):
    sep = "=" * 58
    logger.info(f"\n{sep}\nnnUNet — TEST SET RESULTS\n{sep}")
    logger.info(f"\n{'Region':<6} {'Dice mean':>10} {'± std':>8} "
                f"{'Median':>8} {'IoU':>8} {'HD95':>8} {'AUC':>8}")
    logger.info("-" * 58)
    for region in REGION_NAMES:
        sub  = df[df["region"]==region]
        dice = sub["dice"].dropna(); iou_ = sub["iou"].dropna(); hd_ = sub["hd95"].dropna()
        auc  = aucs.get(region, float("nan"))
        logger.info(f"{region:<6} {dice.mean():>10.4f} {dice.std():>8.4f} "
                    f"{dice.median():>8.4f} {iou_.mean():>8.4f} "
                    f"{hd_.mean():>8.2f} {auc:>8.4f}")
    logger.info(f"\n{'Region':<6} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}")
    logger.info("-" * 42)
    for _, row in cls_df.iterrows():
        logger.info(f"{row['region']:<6} {row['precision']:>10.4f} "
                    f"{row['recall']:>8.4f} {row['f1']:>8.4f} {row['accuracy']:>10.4f}")
    logger.info(sep)


# ===========================================================================
# 10. MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT),
                   help="Path to nnUNet checkpoint_best.pth")
    p.add_argument("--n_viz", type=int, default=N_VIZ_CASES)
    return p.parse_args()


def main():
    args      = parse_args()
    ckpt_path = Path(args.checkpoint)

    logger.info("=" * 58)
    logger.info("nnUNet — Evaluation pipeline (direct checkpoint)")
    logger.info("=" * 58)
    logger.info(f"Checkpoint: {ckpt_path}")

    all_cases = discover_cases(DATA_ROOT)
    splits    = build_splits(all_cases)
    test_dirs = get_case_paths_for_split(splits["test"], DATA_ROOT)
    logger.info(f"Test cases: {len(test_dirs)}")

    network, ckpt, plans = load_nnunet_model(ckpt_path)

    df, probs_all, gts_all = evaluate_all_cases(network, test_dirs, plans)

    if df.empty:
        logger.error("No cases evaluated successfully.")
        return

    df.to_csv(OUT_DIR / "metrics_per_case.csv", index=False)
    summary = (df.groupby("region")["dice"]
               .agg(mean_dice="mean", std_dice="std",
                    median_dice="median", min_dice="min", max_dice="max")
               .reset_index())
    summary.to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    cls_df = classification_metrics(df)
    cls_df.to_csv(OUT_DIR / "classification_metrics.csv", index=False)
    logger.info(f"Saved CSVs to {OUT_DIR}")

    logger.info("Computing ROC-AUC …")
    aucs = compute_and_plot_roc(probs_all, gts_all)

    summary_json = {
        region: {
            "mean_dice":   round(float(df[df["region"]==region]["dice"].mean()), 4),
            "std_dice":    round(float(df[df["region"]==region]["dice"].std()),  4),
            "median_dice": round(float(df[df["region"]==region]["dice"].median()),4),
            "mean_iou":    round(float(df[df["region"]==region]["iou"].mean()),  4),
            "mean_hd95":   round(float(df[df["region"]==region]["hd95"].mean()), 4),
            "roc_auc":     aucs.get(region, None),
        }
        for region in REGION_NAMES
    }
    with open(OUT_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    logger.info("Generating plots …")
    plot_dice_boxplot(df)
    plot_dice_bar(df)
    plot_per_case_dice(df)
    plot_training_curve(ckpt)

    logger.info(f"Generating overlays (best/worst {args.n_viz}) …")
    plot_overlays(network, test_dirs, df, plans, args.n_viz)

    print_summary(df, cls_df, aucs)
    logger.info(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()