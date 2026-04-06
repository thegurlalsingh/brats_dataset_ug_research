"""
config.py — single source of truth for the entire pipeline.
All paths, hyperparameters, and BraTS label conventions live here.
No other file should hardcode any of these values.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PATHS
# ---------------------------------------------------------------------------

# Root of your BraTS-GLI folder (contains BraTS-GLI-XXXXX-XXX subfolders)
DATA_ROOT = Path(r"C:\Users\Gurlal-Stu\Downloads\new_code\data")

# All outputs go under this folder (created automatically if missing)
OUTPUT_ROOT = Path(r"C:\Users\Gurlal-Stu\Downloads\new_code\outputs")

CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"   # saved model weights
RESULTS_DIR    = OUTPUT_ROOT / "results"        # metrics CSVs, plots
XAI_DIR        = OUTPUT_ROOT / "xai"           # CAM maps, mask overlays
NNUNET_RAW     = OUTPUT_ROOT / "nnunet_raw"    # nnUNet dataset conversion
NNUNET_PREPROCESSED = OUTPUT_ROOT / "nnunet_preprocessed"
NNUNET_RESULTS = OUTPUT_ROOT / "nnunet_results"

# Split cache — reproducible across runs
SPLIT_CACHE = OUTPUT_ROOT / "splits.json"

# Create all output directories on import
for _d in [CHECKPOINT_DIR, RESULTS_DIR, XAI_DIR,
           NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS]:
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 2. MODALITY ORDER  (THE mistake that killed your Dice — locked here)
# ---------------------------------------------------------------------------
# BraTS 2023 GLI has exactly these 4 modalities per case.
# The suffix in the filename tells you which is which.
# We ALWAYS load them in this fixed order: ch0=t1n, ch1=t1c, ch2=t2w, ch3=t2f
# nnUNet channel IDs must match this order exactly (see train_nnunet.py).

MODALITY_KEYS = ["t1n", "t1c", "t2w", "t2f"]   # file suffix order
MODALITY_LABELS = {
    "t1n": "T1 native",
    "t1c": "T1 contrast",
    "t2w": "T2 weighted",
    "t2f": "T2 FLAIR",
}
NUM_MODALITIES = 4   # input channels to every model


# ---------------------------------------------------------------------------
# 3. LABEL / SEGMENTATION CONVENTION  (the other mistake — fixed here)
# ---------------------------------------------------------------------------
# BraTS 2023 segmentation mask integer values:
#   0 = background
#   1 = necrotic tumor core (NCR)
#   2 = peritumoral edematous / invaded tissue (ED)
#   3 = GD-enhancing tumor (ET)
#
# The three evaluation REGIONS are derived as:
#   WT (whole tumor)      = labels 1 + 2 + 3  → output channel 0
#   TC (tumor core)       = labels 1 + 3       → output channel 1
#   ET (enhancing tumor)  = label  3            → output channel 2
#
# We convert the raw integer mask to a 3-channel binary mask in dataset.py.
# Models predict 3 channels. Dice is computed per channel.

SEG_SUFFIX = "seg"          # filename suffix for the segmentation file

LABEL_NAMES  = {0: "background", 1: "NCR", 2: "ED", 3: "ET_raw"}
REGION_NAMES = ["WT", "TC", "ET"]   # must match output channel order
NUM_CLASSES  = 3                      # number of segmentation output channels


# ---------------------------------------------------------------------------
# 4. TRAIN / VAL / TEST SPLIT
# ---------------------------------------------------------------------------
# Fixed seed — never change after first run so test set stays truly held-out.
# With ~150 patients: 105 train | 22 val | 23 test  (70/15/15 %)

SPLIT_SEED       = 42
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
# TEST_RATIO is implicitly 1 - TRAIN_RATIO - VAL_RATIO = 0.15


# ---------------------------------------------------------------------------
# 5. PREPROCESSING
# ---------------------------------------------------------------------------
# Z-score normalization: subtract mean, divide by std.
# Computed ONLY over the non-zero (brain) voxels of each modality independently.
# This is the correct BraTS normalization — do NOT normalize over the full volume
# (that was one of your earlier mistakes).

NORMALIZE_NONZERO_ONLY = True   # True = mask to brain region before stats

# Spatial resampling — BraTS 2023 is already 1mm isotropic 240x240x155
# We crop/pad to a fixed size for batching. 128^3 fits on A6000 comfortably.
SPATIAL_SIZE = (128, 128, 128)   # (D, H, W) after crop/pad


# ---------------------------------------------------------------------------
# 6. TRAINING HYPERPARAMETERS — MONAI models (SegResNet + Swin UNETR)
# ---------------------------------------------------------------------------

EPOCHS          = 100
BATCH_SIZE      = 1       # 3D volumes are large; 1 per GPU is standard
VAL_INTERVAL    = 2       # validate every N epochs
NUM_WORKERS     = 4       # DataLoader workers (adjust to your CPU count)
PIN_MEMORY      = True

# Optimizer
LR_INITIAL      = 1e-4
WEIGHT_DECAY    = 1e-5

# LR scheduler: cosine annealing down to this floor
LR_MIN          = 1e-6

# Loss: DiceFocalLoss — best for BraTS class imbalance
# lambda_dice and lambda_focal weight the two terms
LOSS_LAMBDA_DICE  = 1.0
LOSS_LAMBDA_FOCAL = 1.0
FOCAL_GAMMA       = 2.0   # focusing parameter for hard examples

# Mixed precision (speeds up A6000 significantly)
USE_AMP = True

# Early stopping patience (in validation intervals)
EARLY_STOP_PATIENCE = 20   # stop if no val Dice improvement for 20 val checks


# ---------------------------------------------------------------------------
# 7. SEGRESNET ARCHITECTURE
# ---------------------------------------------------------------------------

SEGRESNET_CFG = {
    "spatial_dims": 3,
    "in_channels":  NUM_MODALITIES,   # 4
    "out_channels": NUM_CLASSES,       # 3
    "init_filters": 32,
    "blocks_down":  (1, 2, 2, 4),
    "blocks_up":    (1, 1, 1),
    "dropout_prob": 0.2,
}


# ---------------------------------------------------------------------------
# 8. SWIN UNETR ARCHITECTURE
# ---------------------------------------------------------------------------

SWIN_CFG = {
    # "img_size":     SPATIAL_SIZE,      # (128, 128, 128)
    "in_channels":  NUM_MODALITIES,    # 4
    "out_channels": NUM_CLASSES,       # 3
    "feature_size": 48,
    "use_checkpoint": True,            # gradient checkpointing — saves VRAM
}


# ---------------------------------------------------------------------------
# 9. NNUNET SETTINGS
# ---------------------------------------------------------------------------
# nnUNet v2 uses its own preprocessing pipeline internally.
# We only need to convert BraTS data to nnUNet's raw dataset format.
# Dataset ID must be unique (choose any 3-digit number not already used).

NNUNET_DATASET_ID   = 137            # nnUNet dataset identifier
NNUNET_DATASET_NAME = f"Dataset{NNUNET_DATASET_ID:03d}_BraTSGLI"
NNUNET_TRAINER      = "nnUNetTrainer"
NNUNET_CONFIG       = "3d_fullres"   # use full resolution 3D config
NNUNET_FOLD         = 0              # train on fold 0 of nnUNet's 5-fold CV
NNUNET_EPOCHS       = 80            # passed via --num_epochs flag


# ---------------------------------------------------------------------------
# 10. EVALUATION & XAI
# ---------------------------------------------------------------------------

# Sliding window inference settings (MONAI)
SW_ROI_SIZE   = SPATIAL_SIZE   # same as training patch
SW_OVERLAP    = 0.5            # 50% overlap → better boundary prediction

# ForwardCAM: which layer name to hook for SegResNet and Swin UNETR
# (nnUNet uses its own forward-pass CAM wrapper — see xai.py)
SEGRESNET_CAM_LAYER = "convolutions.4"   # last encoder block
SWIN_CAM_LAYER      = "swinViT.layers4"  # deepest Swin stage

# Occlusion XAI patch size (modality-level sensitivity)
OCCLUSION_PATCH = (16, 16, 16)

# Number of worst / best cases to visualise in the report
N_VIZ_CASES = 5

# ROC-AUC: flatten 3D predictions to 1D per region for sklearn
ROC_REGIONS = REGION_NAMES   # ["WT", "TC", "ET"]


# ---------------------------------------------------------------------------
# 11. DEVICE
# ---------------------------------------------------------------------------
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 12. REPRODUCIBILITY
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42


# ---------------------------------------------------------------------------
# SANITY CHECK — run this file directly to verify paths exist
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== config.py sanity check ===")
    print(f"DATA_ROOT   : {DATA_ROOT}")
    print(f"  exists    : {DATA_ROOT.exists()}")
    cases = [d for d in DATA_ROOT.iterdir() if d.is_dir()] if DATA_ROOT.exists() else []
    print(f"  cases found: {len(cases)}")
    if cases:
        sample = cases[0]
        print(f"  sample case : {sample.name}")
        files = list(sample.glob("*.nii.gz"))
        print(f"  files in it : {[f.name for f in files]}")
    print(f"\nOUTPUT_ROOT : {OUTPUT_ROOT}")
    print(f"DEVICE      : {DEVICE}")
    print(f"MODALITIES  : {MODALITY_KEYS}")
    print(f"REGIONS     : {REGION_NAMES}")
    print(f"SPLIT SEED  : {SPLIT_SEED}")
    print(f"EPOCHS      : {EPOCHS}")
    print("\nAll output dirs created. config.py OK.")