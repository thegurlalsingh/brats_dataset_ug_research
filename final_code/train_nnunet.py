"""
train_nnunet.py — nnUNet v2 data conversion + training wrapper.

nnUNet manages its own preprocessing, patch size, and normalization internally.
This script does three things in order:
  Step 1 — Convert BraTS data to nnUNet raw dataset format (imagesTr/labelsTr/imagesTs)
  Step 2 — Run nnUNet dataset fingerprint + experiment planning + preprocessing
  Step 3 — Train nnUNet on fold 0 for NNUNET_EPOCHS epochs

The train/val split used is the SAME splits.json produced by dataset.py so
the test set is never touched. nnUNet's own internal cross-validation runs
on our training cases only.

Run with:
    python train_nnunet.py

Requirements:
    pip install nnunetv2
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR,
    DATA_ROOT,
    MODALITY_KEYS,
    NNUNET_CONFIG,
    NNUNET_DATASET_ID,
    NNUNET_DATASET_NAME,
    NNUNET_EPOCHS,
    NNUNET_FOLD,
    NNUNET_PREPROCESSED,
    NNUNET_RAW,
    NNUNET_RESULTS,
    NNUNET_TRAINER,
    OUTPUT_ROOT,
    REGION_NAMES,
    RESULTS_DIR,
    SEG_SUFFIX,
    SPLIT_CACHE,
)
from dataset import build_splits, discover_cases, get_case_paths_for_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# nnUNet v2 expects three environment variables pointing to its data folders.
# We set them to our output directories so everything stays under outputs/.
# ---------------------------------------------------------------------------
os.environ["nnUNet_raw"]          = str(NNUNET_RAW)
os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
os.environ["nnUNet_results"]      = str(NNUNET_RESULTS)

# Path where nnUNet will write this dataset
DATASET_RAW_DIR = NNUNET_RAW / NNUNET_DATASET_NAME
IMAGES_TR_DIR   = DATASET_RAW_DIR / "imagesTr"
LABELS_TR_DIR   = DATASET_RAW_DIR / "labelsTr"
IMAGES_TS_DIR   = DATASET_RAW_DIR / "imagesTs"
LABELS_TS_DIR   = DATASET_RAW_DIR / "labelsTs"

# nnUNet log file written during training — we tail it for tqdm
NNUNET_LOG = NNUNET_RESULTS / NNUNET_DATASET_NAME / NNUNET_TRAINER / \
             f"{NNUNET_CONFIG}__nnUNetPlans" / f"fold_{NNUNET_FOLD}" / "training_log.txt"


# ===========================================================================
# STEP 1 — CONVERT DATA TO nnUNet RAW FORMAT
# ===========================================================================

def _copy_modalities(case_dir: Path, dest_dir: Path, case_id: str) -> None:
    """
    Copy the 4 modality files for one case into dest_dir using nnUNet naming.

    nnUNet v2 file naming:
        {case_id}_{channel_id:04d}.nii.gz
    Channel order matches MODALITY_KEYS = [t1n, t1c, t2w, t2f]
    so channel 0000=t1n, 0001=t1c, 0002=t2w, 0003=t2f.
    This is the SAME order locked in config.py — never reorder here.
    """
    for ch_idx, suffix in enumerate(MODALITY_KEYS):
        src  = case_dir / f"{case_dir.name}-{suffix}.nii.gz"
        dst  = dest_dir / f"{case_id}_{ch_idx:04d}.nii.gz"
        if not dst.exists():
            shutil.copy2(src, dst)


def _convert_label(case_dir: Path, dest_dir: Path, case_id: str) -> None:
    """
    Copy the segmentation label for one case.

    nnUNet expects RAW integer labels {0,1,2,3} — NOT the 3-channel binary
    masks we use for MONAI models. So we copy the original seg file directly.
    nnUNet handles the WT/TC/ET region logic internally through its region
    loss configuration in the dataset.json.
    """
    src = case_dir / f"{case_dir.name}-{SEG_SUFFIX}.nii.gz"
    dst = dest_dir / f"{case_id}.nii.gz"
    if not dst.exists():
        shutil.copy2(src, dst)


def build_dataset_json() -> dict:
    """
    Build the dataset.json that tells nnUNet about channels, labels, and regions.

    For BraTS 2023 the labels block uses the 'regions' key which instructs
    nnUNet to use region-based training (predicting WT, TC, ET simultaneously
    using overlapping binary masks), matching the official BraTS benchmark setup.
    """
    return {
        "channel_names": {
            "0": "T1n",
            "1": "T1c",
            "2": "T2w",
            "3": "T2f",
        },
        "labels": {
            "background": 0,
            "whole_tumor": [1, 2, 3],    # WT = NCR + ED + ET
            "tumor_core":  [1, 3],       # TC = NCR + ET
            "enhancing":   [3],          # ET = ET only
        },
        "regions_class_order": [1, 2, 3],
        "numTraining": None,             # filled in after conversion
        "file_ending": ".nii.gz",
        "name": NNUNET_DATASET_NAME,
        "description": "BraTS 2023 GLI subset — 150 cases",
        "reference": "BraTS 2023",
        "licence": "CC-BY 4.0",
        "release": "1.0",
        "overwrite_image_reader_writer": "NibabelIOWithReorient",
    }


def convert_dataset(
    train_dirs: list,
    val_dirs:   list,
    test_dirs:  list,
) -> None:
    """
    Convert BraTS cases to nnUNet raw format.

    nnUNet treats train+val together as 'training' (it does its own CV).
    Test cases go to imagesTs for later inference.
    """
    if DATASET_RAW_DIR.exists() and (IMAGES_TR_DIR / f"{train_dirs[0].name}_0000.nii.gz").exists():
        logger.info("nnUNet raw dataset already exists — skipping conversion.")
        return

    logger.info(f"Converting data to nnUNet format → {DATASET_RAW_DIR}")

    for d in [IMAGES_TR_DIR, LABELS_TR_DIR, IMAGES_TS_DIR, LABELS_TS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Train + val → imagesTr / labelsTr
    tr_cases = train_dirs + val_dirs
    pbar = tqdm(tr_cases, desc="Converting train+val", unit="case")
    for case_dir in pbar:
        pbar.set_postfix(case=case_dir.name)
        _copy_modalities(case_dir, IMAGES_TR_DIR, case_dir.name)
        _convert_label(case_dir, LABELS_TR_DIR,   case_dir.name)

    # Test → imagesTs / labelsTs
    pbar = tqdm(test_dirs, desc="Converting test     ", unit="case")
    for case_dir in pbar:
        pbar.set_postfix(case=case_dir.name)
        _copy_modalities(case_dir, IMAGES_TS_DIR, case_dir.name)
        _convert_label(case_dir, LABELS_TS_DIR,   case_dir.name)

    # Write dataset.json
    ds_json = build_dataset_json()
    ds_json["numTraining"] = len(tr_cases)
    with open(DATASET_RAW_DIR / "dataset.json", "w") as f:
        json.dump(ds_json, f, indent=2)

    logger.info(
        f"Conversion complete — "
        f"{len(tr_cases)} training cases, {len(test_dirs)} test cases."
    )


# ===========================================================================
# STEP 2 — FINGERPRINT + PLAN + PREPROCESS
# ===========================================================================

def run_command(cmd: list, step_name: str) -> None:
    """Run a shell command, stream output live, raise on failure."""
    logger.info(f"Running: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ.copy(),
    )

    # Stream output line by line so the user sees progress
    for line in process.stdout:
        line = line.rstrip()
        if line:
            logger.info(f"  [nnUNet] {line}")

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"{step_name} failed with return code {process.returncode}.\n"
            "Check the output above for the error message."
        )
    logger.info(f"{step_name} — done.")


def plan_and_preprocess() -> None:
    """
    Run nnUNet dataset fingerprinting, experiment planning, and preprocessing.
    This only needs to run once. If the preprocessed folder already has content
    it is skipped.
    """
    preproc_check = NNUNET_PREPROCESSED / NNUNET_DATASET_NAME
    if preproc_check.exists() and any(preproc_check.iterdir()):
        logger.info("Preprocessed data already exists — skipping plan+preprocess.")
        return

    logger.info("=== Step 2: Plan and preprocess ===")

    # nnUNetv2_plan_and_preprocess replaces the old three-step pipeline
    run_command(
        [
            sys.executable, "-m", "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
            "-d", str(NNUNET_DATASET_ID),
            "-c", NNUNET_CONFIG,
            "--verify_dataset_integrity",
            "-np", "4",   # number of preprocessing workers
        ],
        "Plan and preprocess",
    )


# ===========================================================================
# STEP 3 — TRAINING WITH LIVE TQDM PROGRESS
# ===========================================================================

def _parse_epoch_from_log(log_path: Path) -> tuple:
    """
    Parse the most recent epoch number and dice score from nnUNet's training log.
    nnUNet v2 writes lines like:
        Epoch X  train_loss -0.1234  val_loss -0.5678  ...
    or pseudo_dice lines. Returns (epoch, pseudo_dice) or (None, None).
    """
    if not log_path.exists():
        return None, None

    epoch = None
    dice  = None
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                # Match "Epoch X" at start of line
                m = re.search(r"Epoch\s+(\d+)", line)
                if m:
                    epoch = int(m.group(1))
                # Match pseudo_dice or mean_fg_dice value
                m2 = re.search(r"(?:pseudo_dice|mean_fg_dice)[^\d]*([\d.]+)", line)
                if m2:
                    try:
                        dice = float(m2.group(1))
                    except ValueError:
                        pass
    except Exception:
        pass

    return epoch, dice


def _tail_log_to_tqdm(log_path: Path, total_epochs: int, stop_event: threading.Event) -> None:
    """
    Background thread: watches nnUNet's log file and updates a tqdm bar.
    Runs until stop_event is set.
    """
    pbar = tqdm(total=total_epochs, desc="nnUNet training", unit="epoch")
    last_epoch = 0

    while not stop_event.is_set():
        epoch, dice = _parse_epoch_from_log(log_path)
        if epoch is not None and epoch > last_epoch:
            pbar.update(epoch - last_epoch)
            last_epoch = epoch
            postfix = {"epoch": epoch}
            if dice is not None:
                postfix["pseudo_dice"] = f"{dice:.4f}"
            pbar.set_postfix(postfix)
        time.sleep(2)   # poll every 2 seconds

    # Final update to 100%
    if last_epoch < total_epochs:
        pbar.update(total_epochs - last_epoch)
    pbar.close()



def _install_custom_trainer() -> str:
    """
    nnUNet v2 discovers trainer classes by scanning its own installed package
    directory — specifically nnunetv2/training/nnUNetTrainer/*.py.
    PYTHONPATH injection is ignored. The only reliable way to use a custom
    trainer is to copy its file into that directory.

    This function:
      1. Finds the nnUNet trainer directory from the installed package.
      2. Copies nnunet_trainer_custom.py there if not already present.
      3. Returns the trainer class name to pass to -tr.

    If copying fails (permissions etc.) falls back to default nnUNetTrainer
    and logs a warning so training still proceeds (just at 1000 epochs).
    """
    import importlib
    TRAINER_CLASS = "nnUNetTrainerCustom"
    TRAINER_FILE  = Path(__file__).parent / "nnunet_trainer_custom.py"

    try:
        import nnunetv2
        nnunet_trainer_dir = (
            Path(nnunetv2.__file__).parent
            / "training" / "nnUNetTrainer"
        )
        dest = nnunet_trainer_dir / "nnUNetTrainerCustom.py"

        if not dest.exists():
            shutil.copy2(TRAINER_FILE, dest)
            logger.info(f"Installed custom trainer → {dest}")
        else:
            logger.info(f"Custom trainer already installed at {dest}")

        return TRAINER_CLASS

    except Exception as e:
        logger.warning(
            f"Could not install custom trainer ({e}). "
            f"Falling back to default nnUNetTrainer (1000 epochs). "
            f"To use 80 epochs, manually copy nnunet_trainer_custom.py into: "
            f"<nnunetv2_install>/training/nnUNetTrainer/"
        )
        return "nnUNetTrainer"


def train_nnunet() -> None:
    """
    Launch nnUNet training via subprocess and monitor progress with tqdm.
    """
    logger.info("=== Step 3: Training nnUNet ===")
    logger.info(
        f"Dataset: {NNUNET_DATASET_NAME} | Config: {NNUNET_CONFIG} | "
        f"Fold: {NNUNET_FOLD} | Epochs: {NNUNET_EPOCHS}"
    )

    # Ensure the log directory exists so the tail thread doesn't error
    NNUNET_LOG.parent.mkdir(parents=True, exist_ok=True)

    # ── Install custom trainer into nnUNet's package directory ──────────────
    # nnUNet v2 only scans its own installed package for trainer classes.
    # PYTHONPATH injection does NOT work — it hardcodes the scan path.
    # Solution: copy our trainer file into the nnUNet trainer directory at runtime.
    trainer_name = _install_custom_trainer()

    cmd = [
        sys.executable, "-m", "nnunetv2.run.run_training",
        str(NNUNET_DATASET_ID),
        NNUNET_CONFIG,
        str(NNUNET_FOLD),
        "-tr", trainer_name,
        "--npz",   # save softmax outputs for evaluate.py and xai.py
    ]

    train_env = os.environ.copy()

    # Start tqdm watcher in background thread
    stop_event = threading.Event()
    watcher = threading.Thread(
        target=_tail_log_to_tqdm,
        args=(NNUNET_LOG, NNUNET_EPOCHS, stop_event),
        daemon=True,
    )
    watcher.start()

    raw_log = RESULTS_DIR / "nnunet_train_raw.log"

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=train_env,
        )

        with open(raw_log, "w", encoding="utf-8") as log_f:
            while True:
                try:
                    line = process.stdout.readline()
                except KeyboardInterrupt:
                    # Training is done — nnUNet closed stdout, readline blocks.
                    # Just break and let process.wait() confirm exit code.
                    logger.info("Log reader interrupted — checking process exit.")
                    break
                if line == "" and process.poll() is not None:
                    break
                if line:
                    log_f.write(line)
                    log_f.flush()

        process.wait()

    except KeyboardInterrupt:
        logger.info("Interrupted — training subprocess may still be running.")
        process.wait()

    finally:
        stop_event.set()
        watcher.join(timeout=5)

    rc = process.returncode
    if rc not in (0, None):
        raise RuntimeError(
            f"nnUNet training failed (return code {rc}).\n"
            f"Check raw log: {raw_log}"
        )

    logger.info("nnUNet training complete.")


# ===========================================================================
# STEP 4 — COPY BEST CHECKPOINT TO OUR CHECKPOINTS DIR
# ===========================================================================

def export_best_checkpoint() -> None:
    """
    Copy nnUNet's best checkpoint into our unified checkpoints directory
    so evaluate.py and xai.py can find it alongside the MONAI model checkpoints.
    """
    fold_dir = (
        NNUNET_RESULTS / NNUNET_DATASET_NAME / NNUNET_TRAINER
        / f"{NNUNET_CONFIG}__nnUNetPlans" / f"fold_{NNUNET_FOLD}"
    )

    # nnUNet saves checkpoint_best.pth and checkpoint_final.pth
    for ckpt_name in ["checkpoint_best.pth", "checkpoint_final.pth"]:
        src = fold_dir / ckpt_name
        if src.exists():
            dst = CHECKPOINT_DIR / f"nnunet_{ckpt_name}"
            shutil.copy2(src, dst)
            logger.info(f"Exported: {dst}")

    # Also copy the plans file — needed by evaluate.py to rebuild the model
    plans_src = fold_dir.parent / "nnUNetPlans.json"
    if plans_src.exists():
        shutil.copy2(plans_src, CHECKPOINT_DIR / "nnUNetPlans.json")
        logger.info(f"Exported plans: {CHECKPOINT_DIR / 'nnUNetPlans.json'}")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("nnUNet v2 — BraTS 2023 GLI training pipeline")
    logger.info("=" * 60)

    # Check nnUNet is installed
    try:
        import nnunetv2
        logger.info(f"nnUNetv2 version: {nnunetv2.__version__}")
    except ImportError:
        logger.error(
            "nnunetv2 is not installed.\n"
            "Install with:  pip install nnunetv2"
        )
        sys.exit(1)

    # Load the same split used by all other models
    logger.info("Loading split from dataset.py …")
    all_cases  = discover_cases(DATA_ROOT)
    splits     = build_splits(all_cases)
    train_dirs = get_case_paths_for_split(splits["train"], DATA_ROOT)
    val_dirs   = get_case_paths_for_split(splits["val"],   DATA_ROOT)
    test_dirs  = get_case_paths_for_split(splits["test"],  DATA_ROOT)

    logger.info(
        f"Split — train: {len(train_dirs)} | val: {len(val_dirs)} | test: {len(test_dirs)}"
    )

    # ── Step 1: Convert ──────────────────────────────────────────────────────
    logger.info("=== Step 1: Convert dataset ===")
    convert_dataset(train_dirs, val_dirs, test_dirs)

    # ── Step 2: Plan + preprocess ────────────────────────────────────────────
    plan_and_preprocess()

    # ── Step 3: Train ────────────────────────────────────────────────────────
    train_nnunet()

    # ── Step 4: Export checkpoint ────────────────────────────────────────────
    logger.info("=== Step 4: Export checkpoint ===")
    export_best_checkpoint()

    logger.info("=" * 60)
    logger.info("nnUNet pipeline complete.")
    logger.info(f"  Raw data     : {DATASET_RAW_DIR}")
    logger.info(f"  Preprocessed : {NNUNET_PREPROCESSED / NNUNET_DATASET_NAME}")
    logger.info(f"  Results      : {NNUNET_RESULTS / NNUNET_DATASET_NAME}")
    logger.info(f"  Checkpoints  : {CHECKPOINT_DIR}")
    logger.info("=" * 60)
    logger.info("Next step: run evaluate.py to get Dice/ROC-AUC on the test set.")


if __name__ == "__main__":
    main()