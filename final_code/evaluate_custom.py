"""
evaluate_custom.py — Evaluation script for the custom BraTS 2.5D model.

Loads the best checkpoint saved by train_custom.py and evaluates on the
held-out test set. Aggregates per-slice predictions back to full 3D volumes
before computing metrics so results are comparable to nnUNet / SegResNet.

Outputs saved to config.RESULTS_DIR / "custom":
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
  training_curve.png           — loss / Dice history from training JSON

Usage:
    python evaluate_custom.py
    python evaluate_custom.py --checkpoint path/to/custom_best.pth --n_viz 3
"""

import argparse
import json
import logging
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
    REGION_NAMES, RESULTS_DIR, CHECKPOINT_DIR,
    MODALITY_KEYS, SEG_SUFFIX, NUM_MODALITIES,
)
from dataset import (
    build_splits, convert_labels, discover_cases,
    get_case_paths_for_split,
)
from custom_model_sanity_check import BraTS25DDataset, CustomBraTSModel

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR     = RESULTS_DIR / "custom"
ROC_DIR     = OUT_DIR / "roc_curves"
MASK_DIR    = OUT_DIR / "pred_masks"
OVERLAY_DIR = OUT_DIR / "overlay_viz"
for _d in [ROC_DIR, MASK_DIR, OVERLAY_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_CKPT    = CHECKPOINT_DIR / "custom_best.pth"
DEFAULT_HISTORY = RESULTS_DIR    / "custom_history.json"

REGION_COLORS = {"WT": "#E84C4C", "TC": "#4CAF50", "ET": "#4C8BE8"}

# Model hyper-params — must match train_custom.py
K          = 3
TARGET_HW  = (192, 192)
BASE_CH    = 64
PATCH_SIZE = 8


# ===========================================================================
# 1. MODEL LOADING
# ===========================================================================

def load_custom_model(ckpt_path: Path) -> Tuple[torch.nn.Module, dict]:
    logger.info(f"Loading custom model checkpoint: {ckpt_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    logger.info(f"  Saved at epoch : {ckpt.get('epoch', '?')}")
    logger.info(f"  Best val Dice  : {ckpt.get('best_dice', '?')}")

    model = CustomBraTSModel(k=K, base_channels=BASE_CH, patch_size=PATCH_SIZE)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(DEVICE)
    model.eval()
    logger.info("  Custom model loaded and ready.")
    return model, ckpt


# ===========================================================================
# 2. 3D INFERENCE via slice aggregation
# ===========================================================================

def _load_and_normalize(case_dir: Path) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load all 4 modalities + seg for a case. Returns (vols, seg)."""
    case_id = case_dir.name
    vols = []
    for suffix in MODALITY_KEYS:
        v = nib.load(str(case_dir / f"{case_id}-{suffix}.nii.gz")).get_fdata(dtype=np.float32)
        mask = v > 0
        if mask.any():
            v = (v - v[mask].mean()) / max(v[mask].std(), 1e-8)
        vols.append(v)
    seg = nib.load(str(case_dir / f"{case_id}-{SEG_SUFFIX}.nii.gz")).get_fdata()
    return vols, seg.astype(np.float32)


@torch.no_grad()
def predict_volume(model: torch.nn.Module, case_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the 2.5D model on every axial slice of a case and accumulate
    soft probability maps into a full (3, H, W, D) volume.

    Strategy:
      - For each slice z, extract k adjacent slices → (1, 4*k, H, W) input
      - Run model, take out["final"] → (1, 3, H, W) probabilities for that z
      - Accumulate into prob_volume[:, :, :, z]  (avg over multiple hits)

    Returns:
        pred_bin  : (3, H, W, D) binary uint8
        pred_prob : (3, H, W, D) float32 probabilities
        gt_3ch    : (3, H, W, D) binary float32 ground truth
    """
    vols, seg = _load_and_normalize(case_dir)

    H_orig, W_orig, D = vols[0].shape
    pad = K // 2

    # Accumulator in nibabel (H, W, D) convention
    prob_accum = np.zeros((3, H_orig, W_orig, D), dtype=np.float32)

    for z in range(D):
        slices = []
        for dz in range(-pad, pad + 1):
            iz = int(np.clip(z + dz, 0, D - 1))
            for vol in vols:
                slices.append(vol[:, :, iz])   # (H, W)

        x = np.stack(slices, axis=0).astype(np.float32)          # (4*k, H, W)
        x_t = torch.from_numpy(x).unsqueeze(0)                    # (1, 4*k, H, W)
        x_t = F.interpolate(x_t, size=TARGET_HW, mode='bilinear', align_corners=False)
        x_t = x_t.to(DEVICE)

        with torch.amp.autocast("cuda", enabled=True):
            out = model(x_t)

        # out["final"]: (1, 3, Ht, Wt) — resize back to original H, W
        logits = out["final"]                                      # (1, 3, Ht, Wt)
        if logits.shape[-2:] != (H_orig, W_orig):
            logits = F.interpolate(
                logits, size=(H_orig, W_orig),
                mode='bilinear', align_corners=False,
            )
        prob = torch.sigmoid(logits[0]).cpu().float().numpy()      # (3, H, W)
        prob_accum[:, :, :, z] = prob

    pred_bin = (prob_accum > 0.5).astype(np.float32)

    # Ground-truth in same (3, H, W, D) layout
    wt = np.isin(seg, [1, 2, 3]).astype(np.float32)
    tc = np.isin(seg, [1, 3]).astype(np.float32)
    et = (seg == 3).astype(np.float32)
    gt_3ch = np.stack([wt, tc, et], axis=0)                       # (3, H, W, D)

    return pred_bin, prob_accum, gt_3ch


# ===========================================================================
# 3. METRICS
# ===========================================================================

def dice_coeff(pred, gt, smooth=1e-5):
    p, g = pred.astype(bool), gt.astype(bool)
    return float((2*(p&g).sum() + smooth) / (p.sum() + g.sum() + smooth))

def iou_score(pred, gt, smooth=1e-5):
    p, g = pred.astype(bool), gt.astype(bool)
    return float(((p&g).sum() + smooth) / ((p|g).sum() + smooth))

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
        r: {
            "dice": dice_coeff(pred_bin[i], gt_3ch[i]),
            "iou":  iou_score(pred_bin[i],  gt_3ch[i]),
            "hd95": hausdorff95(pred_bin[i], gt_3ch[i]),
        }
        for i, r in enumerate(REGION_NAMES)
    }


# ===========================================================================
# 4. EVALUATION LOOP
# ===========================================================================

def evaluate_all_cases(
    model:     torch.nn.Module,
    test_dirs: List[Path],
) -> Tuple[pd.DataFrame, Dict, Dict]:
    rows      = []
    probs_all = {r: [] for r in REGION_NAMES}
    gts_all   = {r: [] for r in REGION_NAMES}

    pbar = tqdm(test_dirs, desc="Inference + metrics", unit="case")
    for case_dir in pbar:
        case_id = case_dir.name
        pbar.set_postfix(case=case_id[-14:])
        try:
            pred_bin, prob_vol, gt_3ch = predict_volume(model, case_dir)
            metrics = compute_all_metrics(pred_bin, gt_3ch)

            for region in REGION_NAMES:
                m = metrics[region]
                rows.append({
                    "case_id": case_id, "region": region,
                    "dice": m["dice"], "iou": m["iou"], "hd95": m["hd95"],
                })

            # Sub-sample for ROC (keep memory manageable)
            for i, region in enumerate(REGION_NAMES):
                p_flat = prob_vol[i].ravel()
                g_flat = gt_3ch[i].ravel().astype(np.uint8)
                if len(p_flat) > 50_000:
                    idx = np.random.choice(len(p_flat), 50_000, replace=False)
                    p_flat, g_flat = p_flat[idx], g_flat[idx]
                probs_all[region].append(p_flat)
                gts_all[region].append(g_flat)

            _save_nifti_mask(pred_bin, case_dir)

        except Exception as e:
            logger.warning(f"  Skipping {case_id}: {e}", exc_info=True)

    return pd.DataFrame(rows), probs_all, gts_all


def _save_nifti_mask(pred_bin: np.ndarray, case_dir: Path):
    """Save predicted label map as NIfTI in nibabel (H, W, D) layout."""
    try:
        case_id = case_dir.name
        ref     = nib.load(str(case_dir / f"{case_id}-{SEG_SUFFIX}.nii.gz"))
        # pred_bin: (3, H, W, D) → collapse to label map (H, W, D)
        label   = np.zeros(pred_bin.shape[1:], dtype=np.uint8)
        label[pred_bin[0] > 0.5] = 1   # WT
        label[pred_bin[1] > 0.5] = 2   # TC
        label[pred_bin[2] > 0.5] = 3   # ET
        nib.save(
            nib.Nifti1Image(label, ref.affine, ref.header),
            str(MASK_DIR / f"{case_id}_custom_pred.nii.gz"),
        )
    except Exception as e:
        logger.debug(f"Mask save failed for {case_dir.name}: {e}")


# ===========================================================================
# 5. CLASSIFICATION METRICS
# ===========================================================================

def classification_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region in REGION_NAMES:
        sub    = df[df["region"] == region]["dice"].values
        y_true = (sub >  0.0).astype(int)
        y_pred = (sub >= 0.5).astype(int)
        if y_true.sum() == 0:
            continue
        rows.append({
            "region":    region,
            "n_cases":   len(sub),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true,    y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_true,        y_pred, zero_division=0), 4),
            "accuracy":  round(accuracy_score(y_true,  y_pred),                  4),
            "mean_dice": round(float(sub.mean()), 4),
            "std_dice":  round(float(sub.std()),  4),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 6. ROC-AUC
# ===========================================================================

def compute_and_plot_roc(probs_all: Dict, gts_all: Dict) -> Dict:
    aucs = {}
    for region in REGION_NAMES:
        p     = np.concatenate(probs_all[region])
        g     = np.concatenate(gts_all[region]).astype(int)
        color = REGION_COLORS[region]
        if g.sum() == 0 or (1 - g).sum() == 0:
            aucs[region] = float("nan")
            continue
        try:
            auc         = roc_auc_score(g, p)
            fpr, tpr, _ = roc_curve(g, p)
            aucs[region] = round(float(auc), 4)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color=color, lw=2, label=f"Custom (AUC={auc:.4f})")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
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
# 7. PLOTS
# ===========================================================================

def plot_dice_boxplot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))
    data   = [df[df["region"] == r]["dice"].dropna().values for r in REGION_NAMES]
    colors = [REGION_COLORS[r] for r in REGION_NAMES]
    bp = ax.boxplot(
        data, patch_artist=True, notch=False,
        medianprops=dict(color="black", linewidth=2.5),
        whiskerprops=dict(linewidth=1.4), capprops=dict(linewidth=1.4),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.set_xticklabels(REGION_NAMES, fontsize=12)
    ax.set_ylabel("Dice coefficient", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Custom model — Dice on test set", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    for i, (vals, _) in enumerate(zip(data, REGION_NAMES), start=1):
        if len(vals):
            ax.text(i, 1.01, f"μ={vals.mean():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dice_boxplot.png", dpi=150); plt.close()
    logger.info(f"Saved: {OUT_DIR / 'dice_boxplot.png'}")


def plot_dice_bar(df: pd.DataFrame):
    means  = [df[df["region"] == r]["dice"].mean() for r in REGION_NAMES]
    stds   = [df[df["region"] == r]["dice"].std()  for r in REGION_NAMES]
    colors = [REGION_COLORS[r] for r in REGION_NAMES]
    x      = np.arange(len(REGION_NAMES))
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                  alpha=0.82, width=0.5, error_kw=dict(linewidth=1.5))
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.015,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(REGION_NAMES, fontsize=12)
    ax.set_ylabel("Mean Dice ± std", fontsize=11); ax.set_ylim(0, 1.15)
    ax.set_title("Custom model — Mean Dice per region", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dice_bar.png", dpi=150); plt.close()
    logger.info(f"Saved: {OUT_DIR / 'dice_bar.png'}")


def plot_per_case_dice(df: pd.DataFrame):
    pivot = df.pivot_table(index="case_id", columns="region", values="dice")
    pivot["mean"] = pivot[REGION_NAMES].mean(axis=1)
    pivot = pivot.sort_values("mean")
    fig, ax = plt.subplots(figsize=(max(10, len(pivot) * 0.4), 5))
    x = np.arange(len(pivot))
    for region in REGION_NAMES:
        ax.plot(x, pivot[region].values, marker="o", markersize=4,
                linewidth=1.2, label=region, color=REGION_COLORS[region])
    ax.set_xticks(x)
    ax.set_xticklabels([c[-12:] for c in pivot.index], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Dice", fontsize=11); ax.set_ylim(-0.05, 1.05)
    ax.set_title("Custom model — Per-case Dice (sorted by mean)", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dice_per_case.png", dpi=150); plt.close()
    logger.info(f"Saved: {OUT_DIR / 'dice_per_case.png'}")


def plot_training_curve(history_path: Path):
    """Reload the JSON history saved by train_custom.py and plot it."""
    if not history_path.exists():
        logger.info(f"No training history found at {history_path} — skipping curve.")
        return
    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history.get("train_loss", []), label="train loss", color="#4C8BE8")
    if history.get("val_loss"):
        axes[0].plot(history["val_loss"], label="val loss", color="#E84C4C")
    axes[0].set_title("Loss curve", fontsize=12)
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    colors = list(REGION_COLORS.values())
    for region, key, color in zip(
        REGION_NAMES,
        ["val_dice_wt", "val_dice_tc", "val_dice_et"],
        colors,
    ):
        if history.get(key):
            axes[1].plot(history[key], label=f"Dice {region}", color=color)
    axes[1].set_title("Validation Dice", fontsize=12)
    axes[1].set_xlabel("Val step"); axes[1].set_ylim(0, 1)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Custom model — Training history", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_curve.png", dpi=150); plt.close()
    logger.info(f"Saved: {OUT_DIR / 'training_curve.png'}")


def plot_overlays(model: torch.nn.Module, test_dirs: List[Path],
                  df: pd.DataFrame, n_viz: int):
    mean_dice = df.groupby("case_id")["dice"].mean().reset_index().sort_values("dice")
    worst_ids = mean_dice.head(n_viz)["case_id"].tolist()
    best_ids  = mean_dice.tail(n_viz)["case_id"].tolist()
    dir_map   = {cd.name: cd for cd in test_dirs}

    for case_id, tag in [(c, "worst") for c in worst_ids] + [(c, "best") for c in best_ids]:
        if case_id not in dir_map:
            continue
        case_dir = dir_map[case_id]
        try:
            pred_bin, _, gt_3ch = predict_volume(model, case_dir)
            # Find most-informative axial slice (max WT foreground)
            # gt_3ch: (3, H, W, D) → sum over H,W for each z
            z = int(gt_3ch[0].sum(axis=(0, 1)).argmax())

            flair = nib.load(
                str(case_dir / f"{case_id}-t2f.nii.gz")
            ).get_fdata(dtype=np.float32)             # (H, W, D)
            bg      = flair[:, :, min(z, flair.shape[2] - 1)]
            bg_norm = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)

            fig, axes = plt.subplots(2, 3, figsize=(13, 8))
            for col, region in enumerate(REGION_NAMES):
                for row, (mask, label) in enumerate([
                    (gt_3ch[col, :, :, z],   "Ground truth"),
                    (pred_bin[col, :, :, z], "Custom pred"),
                ]):
                    ax = axes[row][col]
                    ax.imshow(bg_norm.T, cmap="gray", origin="lower", aspect="auto")
                    if mask.any():
                        ax.contour(mask.T, levels=[0.5],
                                   colors=[REGION_COLORS[region]], linewidths=1.8)
                    ax.set_title(f"{region} — {label}", fontsize=9)
                    ax.axis("off")

            case_dice = {
                r: df[(df["case_id"] == case_id) & (df["region"] == r)]["dice"].values
                for r in REGION_NAMES
            }
            dice_str = "  ".join([
                f"{r}={v[0]:.3f}" if len(v) else f"{r}=N/A"
                for r, v in case_dice.items()
            ])
            mean_d = mean_dice[mean_dice["case_id"] == case_id]["dice"].values[0]
            plt.suptitle(
                f"Custom | {case_id} | {tag.upper()} | mean={mean_d:.3f}\n{dice_str}",
                fontsize=10,
            )
            plt.tight_layout()
            plt.savefig(OVERLAY_DIR / f"{tag}_{case_id}.png", dpi=130, bbox_inches="tight")
            plt.close()
            logger.info(f"  Saved overlay: {tag}_{case_id}.png")
        except Exception as e:
            logger.warning(f"  Overlay failed for {case_id}: {e}")


# ===========================================================================
# 8. SUMMARY PRINT
# ===========================================================================

def print_summary(df: pd.DataFrame, cls_df: pd.DataFrame, aucs: Dict):
    sep = "=" * 58
    logger.info(f"\n{sep}\nCustom model — TEST SET RESULTS\n{sep}")
    logger.info(
        f"\n{'Region':<6} {'Dice mean':>10} {'± std':>8} "
        f"{'Median':>8} {'IoU':>8} {'HD95':>8} {'AUC':>8}"
    )
    logger.info("-" * 58)
    for region in REGION_NAMES:
        sub  = df[df["region"] == region]
        dice = sub["dice"].dropna()
        iou_ = sub["iou"].dropna()
        hd_  = sub["hd95"].dropna()
        auc  = aucs.get(region, float("nan"))
        logger.info(
            f"{region:<6} {dice.mean():>10.4f} {dice.std():>8.4f} "
            f"{dice.median():>8.4f} {iou_.mean():>8.4f} "
            f"{hd_.mean():>8.2f} {auc:>8.4f}"
        )
    logger.info(
        f"\n{'Region':<6} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}"
    )
    logger.info("-" * 42)
    for _, row in cls_df.iterrows():
        logger.info(
            f"{row['region']:<6} {row['precision']:>10.4f} "
            f"{row['recall']:>8.4f} {row['f1']:>8.4f} {row['accuracy']:>10.4f}"
        )
    logger.info(sep)


# ===========================================================================
# 9. MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint", type=str, default=str(DEFAULT_CKPT),
        help="Path to custom_best.pth",
    )
    p.add_argument(
        "--history", type=str, default=str(DEFAULT_HISTORY),
        help="Path to custom_history.json (for training curve plot)",
    )
    p.add_argument("--n_viz", type=int, default=N_VIZ_CASES)
    return p.parse_args()


def main():
    args      = parse_args()
    ckpt_path = Path(args.checkpoint)
    hist_path = Path(args.history)

    logger.info("=" * 58)
    logger.info("Custom model — Evaluation pipeline")
    logger.info("=" * 58)
    logger.info(f"Checkpoint : {ckpt_path}")
    logger.info(f"History    : {hist_path}")

    # ── Split ────────────────────────────────────────────────────────────────
    all_cases = discover_cases(DATA_ROOT)
    splits    = build_splits(all_cases)
    test_dirs = get_case_paths_for_split(splits["test"], DATA_ROOT)
    logger.info(f"Test cases : {len(test_dirs)}")

    # ── Model ────────────────────────────────────────────────────────────────
    model, _ = load_custom_model(ckpt_path)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    df, probs_all, gts_all = evaluate_all_cases(model, test_dirs)

    if df.empty:
        logger.error("No cases evaluated successfully.")
        return

    # ── Save CSVs ────────────────────────────────────────────────────────────
    df.to_csv(OUT_DIR / "metrics_per_case.csv", index=False)
    summary = (
        df.groupby("region")["dice"]
        .agg(mean_dice="mean", std_dice="std",
             median_dice="median", min_dice="min", max_dice="max")
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    cls_df = classification_metrics(df)
    cls_df.to_csv(OUT_DIR / "classification_metrics.csv", index=False)
    logger.info(f"Saved CSVs to {OUT_DIR}")

    # ── ROC-AUC ──────────────────────────────────────────────────────────────
    logger.info("Computing ROC-AUC …")
    aucs = compute_and_plot_roc(probs_all, gts_all)

    # ── JSON summary ─────────────────────────────────────────────────────────
    summary_json = {
        region: {
            "mean_dice":   round(float(df[df["region"] == region]["dice"].mean()),   4),
            "std_dice":    round(float(df[df["region"] == region]["dice"].std()),    4),
            "median_dice": round(float(df[df["region"] == region]["dice"].median()), 4),
            "mean_iou":    round(float(df[df["region"] == region]["iou"].mean()),    4),
            "mean_hd95":   round(float(df[df["region"] == region]["hd95"].mean()),   4),
            "roc_auc":     aucs.get(region, None),
        }
        for region in REGION_NAMES
    }
    with open(OUT_DIR / "metrics_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    # ── Plots ────────────────────────────────────────────────────────────────
    logger.info("Generating plots …")
    plot_dice_boxplot(df)
    plot_dice_bar(df)
    plot_per_case_dice(df)
    plot_training_curve(hist_path)

    logger.info(f"Generating overlays (best/worst {args.n_viz}) …")
    plot_overlays(model, test_dirs, df, args.n_viz)

    print_summary(df, cls_df, aucs)
    logger.info(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()