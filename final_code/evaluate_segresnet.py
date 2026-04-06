"""
evaluate_segresnet.py — complete standalone evaluation for SegResNet.

Outputs saved to config.RESULTS_DIR / "segresnet":
  metrics_per_case.csv         — Dice, IoU, HD95 per case per region
  metrics_summary.csv          — mean ± std per region
  classification_metrics.csv   — precision, recall, F1, accuracy per region
  metrics_summary.json         — full summary in JSON
  roc_curves/                  — ROC-AUC PNG per region
  dice_boxplot.png             — Dice distribution per region
  dice_bar.png                 — mean Dice bar chart with error bars
  dice_per_case.png            — per-case Dice sorted by mean
  pred_masks/                  — predicted NIfTI masks for all test cases
  overlay_viz/                 — axial slice overlays GT vs pred (best + worst)
  training_curve.png           — loss + dice curve if checkpoint has history

Usage:
    python evaluate_segresnet.py --checkpoint path/to/segresnet.pth
    python evaluate_segresnet.py --checkpoint path/to/segresnet.pth --n_viz 3
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
    CHECKPOINT_DIR, DATA_ROOT, DEVICE,
    N_VIZ_CASES, NUM_CLASSES, NUM_MODALITIES,
    REGION_NAMES, RESULTS_DIR,
    SEGRESNET_CFG, SPATIAL_SIZE,
    SW_OVERLAP, SW_ROI_SIZE, USE_AMP,
)
from dataset import (
    build_splits, convert_labels, discover_cases,
    get_case_paths_for_split, load_case,
    normalize_image, pad_or_crop,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
OUT_DIR     = RESULTS_DIR / "segresnet"
ROC_DIR     = OUT_DIR / "roc_curves"
MASK_DIR    = OUT_DIR / "pred_masks"
OVERLAY_DIR = OUT_DIR / "overlay_viz"
for _d in [ROC_DIR, MASK_DIR, OVERLAY_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

REGION_COLORS = {"WT": "#E84C4C", "TC": "#4CAF50", "ET": "#4C8BE8"}


# ===========================================================================
# 1. MODEL LOADER
# ===========================================================================

def load_model(ckpt_path: Path) -> Tuple[torch.nn.Module, dict]:
    from monai.networks.nets import SegResNet

    logger.info(f"Loading SegResNet from: {ckpt_path}")
    model = SegResNet(**SEGRESNET_CFG).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Robust state dict extraction
    state = None
    for key in ["model_state", "model", "state_dict",
                "net", "model_state_dict", "network"]:
        if isinstance(ckpt, dict) and key in ckpt:
            state = ckpt[key]
            logger.info(f"  Found state dict under key: '{key}'")
            break

    if state is None:
        if isinstance(ckpt, dict):
            state = ckpt
            logger.info("  Treating entire checkpoint as state dict")
        else:
            raise ValueError(
                f"Cannot extract state dict.\n"
                f"Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
            )

    # Strip DataParallel prefix if present
    if next(iter(state.keys())).startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        logger.info("  Stripped 'module.' prefix")

    model.load_state_dict(state, strict=True)
    model.eval()

    if isinstance(ckpt, dict):
        logger.info(
            f"  Epoch: {ckpt.get('epoch', '?')} | "
            f"Best Dice: {ckpt.get('best_dice', '?')}"
        )

    logger.info("  SegResNet loaded successfully.")
    return model, ckpt


# ===========================================================================
# 2. INFERENCE
# ===========================================================================

@torch.no_grad()
def run_inference(
    model:    torch.nn.Module,
    case_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window inference for one test case.

    Returns:
        pred_bin  : (3, D, H, W)  binary float32
        pred_prob : (3, D, H, W)  sigmoid probabilities float32
        gt_3ch    : (3, D, H, W)  ground-truth binary float32
    """
    from monai.inferers import sliding_window_inference

    image, label_raw = load_case(case_dir)
    image_c = pad_or_crop(normalize_image(image), SPATIAL_SIZE)
    gt_3ch  = pad_or_crop(convert_labels(label_raw), SPATIAL_SIZE)

    x = torch.from_numpy(image_c).float().unsqueeze(0).to(DEVICE)

    with torch.amp.autocast("cuda", enabled=USE_AMP):
        logits = sliding_window_inference(
            inputs=x,
            roi_size=SW_ROI_SIZE,
            sw_batch_size=1,
            predictor=model,
            overlap=SW_OVERLAP,
            mode="gaussian",
        )

    prob     = torch.sigmoid(logits[0]).cpu().float().numpy()
    pred_bin = (prob > 0.5).astype(np.float32)

    return pred_bin, prob, gt_3ch


# ===========================================================================
# 3. METRICS
# ===========================================================================

def dice_coeff(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
    p, g  = pred.astype(bool), gt.astype(bool)
    return float((2 * (p & g).sum() + smooth) / (p.sum() + g.sum() + smooth))


def iou_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-5) -> float:
    p, g  = pred.astype(bool), gt.astype(bool)
    return float(((p & g).sum() + smooth) / ((p | g).sum() + smooth))


def hausdorff95(pred: np.ndarray, gt: np.ndarray) -> float:
    try:
        from scipy.ndimage import distance_transform_edt
        p, g = pred.astype(bool), gt.astype(bool)
        if not p.any() and not g.any():
            return 0.0
        if not p.any() or not g.any():
            return float("nan")
        d1 = np.percentile(distance_transform_edt(~p)[g], 95)
        d2 = np.percentile(distance_transform_edt(~g)[p], 95)
        return float(max(d1, d2))
    except Exception:
        return float("nan")


def compute_all_metrics(pred_bin: np.ndarray, gt_3ch: np.ndarray) -> Dict:
    return {
        region: {
            "dice": dice_coeff(pred_bin[i], gt_3ch[i]),
            "iou":  iou_score(pred_bin[i],  gt_3ch[i]),
            "hd95": hausdorff95(pred_bin[i], gt_3ch[i]),
        }
        for i, region in enumerate(REGION_NAMES)
    }


# ===========================================================================
# 4. MAIN EVALUATION LOOP
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
            pred_bin, prob, gt_3ch = run_inference(model, case_dir)
            metrics = compute_all_metrics(pred_bin, gt_3ch)

            for region in REGION_NAMES:
                m = metrics[region]
                rows.append({
                    "case_id": case_id,
                    "region":  region,
                    "dice":    m["dice"],
                    "iou":     m["iou"],
                    "hd95":    m["hd95"],
                })

            for i, region in enumerate(REGION_NAMES):
                p_flat = prob[i].ravel()
                g_flat = gt_3ch[i].ravel().astype(np.uint8)
                if len(p_flat) > 50_000:
                    idx    = np.random.choice(len(p_flat), 50_000, replace=False)
                    p_flat, g_flat = p_flat[idx], g_flat[idx]
                probs_all[region].append(p_flat)
                gts_all[region].append(g_flat)

            _save_nifti_mask(pred_bin, case_dir)

        except Exception as e:
            logger.warning(f"  Skipping {case_id}: {e}")

    return pd.DataFrame(rows), probs_all, gts_all


def _save_nifti_mask(pred_bin: np.ndarray, case_dir: Path) -> None:
    try:
        import nibabel as nib
        case_id = case_dir.name
        ref     = nib.load(str(case_dir / f"{case_id}-seg.nii.gz"))
        label   = np.zeros(pred_bin.shape[1:], dtype=np.uint8)
        label[pred_bin[0] > 0.5] = 1
        label[pred_bin[1] > 0.5] = 2
        label[pred_bin[2] > 0.5] = 3
        nib.save(
            nib.Nifti1Image(label.transpose(1, 2, 0), ref.affine, ref.header),
            str(MASK_DIR / f"{case_id}_segresnet_pred.nii.gz"),
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
            "recall":    round(recall_score(y_true, y_pred,    zero_division=0), 4),
            "f1":        round(f1_score(y_true, y_pred,        zero_division=0), 4),
            "accuracy":  round(accuracy_score(y_true, y_pred),                   4),
            "mean_dice": round(float(sub.mean()), 4),
            "std_dice":  round(float(sub.std()),  4),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# 6. ROC-AUC
# ===========================================================================

def compute_and_plot_roc(
    probs_all: Dict, gts_all: Dict
) -> Dict[str, float]:
    aucs = {}
    for region in REGION_NAMES:
        p = np.concatenate(probs_all[region])
        g = np.concatenate(gts_all[region]).astype(int)
        color = REGION_COLORS[region]

        if g.sum() == 0 or (1 - g).sum() == 0:
            logger.warning(f"ROC skipped for {region} — only one class.")
            aucs[region] = float("nan")
            continue
        try:
            auc          = roc_auc_score(g, p)
            fpr, tpr, _  = roc_curve(g, p)
            aucs[region] = round(float(auc), 4)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"SegResNet (AUC = {auc:.4f})")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
            ax.fill_between(fpr, tpr, alpha=0.08, color=color)
            ax.set_xlabel("False positive rate", fontsize=11)
            ax.set_ylabel("True positive rate",  fontsize=11)
            ax.set_title(f"ROC curve — {region}", fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
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

def plot_dice_boxplot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    data   = [df[df["region"] == r]["dice"].dropna().values for r in REGION_NAMES]
    colors = [REGION_COLORS[r] for r in REGION_NAMES]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2.5),
                    whiskerprops=dict(linewidth=1.4),
                    capprops=dict(linewidth=1.4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels(REGION_NAMES, fontsize=12)
    ax.set_ylabel("Dice coefficient", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("SegResNet — Dice on test set", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    for i, (vals, region) in enumerate(zip(data, REGION_NAMES), start=1):
        if len(vals):
            ax.text(i, 1.01, f"μ={vals.mean():.3f}",
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dice_boxplot.png", dpi=150)
    plt.close()
    logger.info(f"Saved: {OUT_DIR / 'dice_boxplot.png'}")


def plot_dice_bar(df: pd.DataFrame) -> None:
    means  = [df[df["region"] == r]["dice"].mean() for r in REGION_NAMES]
    stds   = [df[df["region"] == r]["dice"].std()  for r in REGION_NAMES]
    colors = [REGION_COLORS[r] for r in REGION_NAMES]
    x      = np.arange(len(REGION_NAMES))
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                  alpha=0.82, width=0.5, error_kw=dict(linewidth=1.5))
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + std + 0.015, f"{mean:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(REGION_NAMES, fontsize=12)
    ax.set_ylabel("Mean Dice ± std", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("SegResNet — Mean Dice per region", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dice_bar.png", dpi=150)
    plt.close()
    logger.info(f"Saved: {OUT_DIR / 'dice_bar.png'}")


def plot_per_case_dice(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="case_id", columns="region", values="dice")
    pivot["mean"] = pivot[REGION_NAMES].mean(axis=1)
    pivot = pivot.sort_values("mean")
    fig, ax = plt.subplots(figsize=(max(10, len(pivot) * 0.4), 5))
    x = np.arange(len(pivot))
    for region in REGION_NAMES:
        ax.plot(x, pivot[region].values, marker="o", markersize=4,
                linewidth=1.2, label=region, color=REGION_COLORS[region])
    ax.set_xticks(x)
    ax.set_xticklabels([c[-12:] for c in pivot.index],
                       rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Dice", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("SegResNet — Per-case Dice (sorted by mean)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dice_per_case.png", dpi=150)
    plt.close()
    logger.info(f"Saved: {OUT_DIR / 'dice_per_case.png'}")


def plot_overlays(
    model:     torch.nn.Module,
    test_dirs: List[Path],
    df:        pd.DataFrame,
    n_viz:     int,
) -> None:
    import nibabel as nib
    mean_dice = df.groupby("case_id")["dice"].mean().reset_index().sort_values("dice")
    worst_ids = mean_dice.head(n_viz)["case_id"].tolist()
    best_ids  = mean_dice.tail(n_viz)["case_id"].tolist()
    dir_map   = {cd.name: cd for cd in test_dirs}

    for case_id, tag in [(c, "worst") for c in worst_ids] + [(c, "best") for c in best_ids]:
        if case_id not in dir_map:
            continue
        case_dir = dir_map[case_id]
        try:
            pred_bin, _, gt_3ch = run_inference(model, case_dir)
            z = int(gt_3ch[0].sum(axis=(1, 2)).argmax())

            flair   = nib.load(
                str(case_dir / f"{case_id}-t2f.nii.gz")
            ).get_fdata(dtype=np.float32)
            bg      = flair[:, :, min(z, flair.shape[2] - 1)]
            bg_norm = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)

            fig, axes = plt.subplots(2, 3, figsize=(13, 8))
            for col, region in enumerate(REGION_NAMES):
                for row, (mask, label) in enumerate([
                    (gt_3ch[col, z],   "Ground truth"),
                    (pred_bin[col, z], "SegResNet pred"),
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
                f"SegResNet | {case_id} | {tag.upper()} | mean={mean_d:.3f}\n{dice_str}",
                fontsize=10,
            )
            plt.tight_layout()
            plt.savefig(OVERLAY_DIR / f"{tag}_{case_id}.png", dpi=130, bbox_inches="tight")
            plt.close()
            logger.info(f"  Saved overlay: {tag}_{case_id}.png")
        except Exception as e:
            logger.warning(f"  Overlay failed for {case_id}: {e}")


def plot_training_curve(ckpt: dict) -> None:
    history = ckpt.get("history", ckpt.get("log", None))
    if history is None:
        logger.info("No training history in checkpoint — skipping curve plot.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="train loss", color="#4C8BE8")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"],   label="val loss",   color="#E84C4C")
    axes[0].set_title("Loss curve", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    for region, color in REGION_COLORS.items():
        key = f"val_dice_{region.lower()}"
        if key in history:
            axes[1].plot(history[key], label=f"Dice {region}", color=color)
    axes[1].set_title("Validation Dice", fontsize=12)
    axes[1].set_xlabel("Val step")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.suptitle("SegResNet — Training history", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_curve.png", dpi=150)
    plt.close()
    logger.info(f"Saved: {OUT_DIR / 'training_curve.png'}")


# ===========================================================================
# 8. SUMMARY PRINT
# ===========================================================================

def print_summary(df: pd.DataFrame, cls_df: pd.DataFrame, aucs: Dict) -> None:
    sep = "=" * 58
    logger.info(f"\n{sep}")
    logger.info("SEGRESNET — TEST SET RESULTS")
    logger.info(sep)
    logger.info(f"\n{'Region':<6} {'Dice mean':>10} {'± std':>8} "
                f"{'Median':>8} {'IoU':>8} {'HD95':>8} {'AUC':>8}")
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
    logger.info(f"\n{'Region':<6} {'Precision':>10} {'Recall':>8} "
                f"{'F1':>8} {'Accuracy':>10}")
    logger.info("-" * 42)
    for _, row in cls_df.iterrows():
        logger.info(
            f"{row['region']:<6} {row['precision']:>10.4f} "
            f"{row['recall']:>8.4f} {row['f1']:>8.4f} "
            f"{row['accuracy']:>10.4f}"
        )
    logger.info(sep)


# ===========================================================================
# 9. MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint", type=str,
        default=str(CHECKPOINT_DIR / "segresnet_best.pth"),
        help="Path to SegResNet checkpoint (.pth)",
    )
    p.add_argument("--n_viz", type=int, default=N_VIZ_CASES)
    return p.parse_args()


def main():
    args      = parse_args()
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    logger.info("=" * 58)
    logger.info("SegResNet — Evaluation pipeline")
    logger.info("=" * 58)

    all_cases = discover_cases(DATA_ROOT)
    splits    = build_splits(all_cases)
    test_dirs = get_case_paths_for_split(splits["test"], DATA_ROOT)
    logger.info(f"Test cases: {len(test_dirs)}")

    model, ckpt = load_model(ckpt_path)

    df, probs_all, gts_all = evaluate_all_cases(model, test_dirs)
    df.to_csv(OUT_DIR / "metrics_per_case.csv", index=False)
    logger.info(f"Saved: {OUT_DIR / 'metrics_per_case.csv'}")

    summary = (
        df.groupby("region")["dice"]
        .agg(mean_dice="mean", std_dice="std",
             median_dice="median", min_dice="min", max_dice="max")
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "metrics_summary.csv", index=False)

    cls_df = classification_metrics(df)
    cls_df.to_csv(OUT_DIR / "classification_metrics.csv", index=False)
    logger.info(f"Saved: {OUT_DIR / 'classification_metrics.csv'}")

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
    logger.info(f"Saved: {OUT_DIR / 'metrics_summary.json'}")

    logger.info("Generating plots …")
    plot_dice_boxplot(df)
    plot_dice_bar(df)
    plot_per_case_dice(df)
    plot_training_curve(ckpt)

    logger.info(f"Generating overlays (best/worst {args.n_viz}) …")
    plot_overlays(model, test_dirs, df, args.n_viz)

    print_summary(df, cls_df, aucs)
    logger.info(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()