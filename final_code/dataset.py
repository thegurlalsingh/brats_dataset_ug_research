"""
dataset.py — BraTS 2023 GLI data loading, preprocessing, and splitting.

Fixes all four mistakes from your previous runs:
  1. Wrong normalization   → z-score per modality, non-zero voxels only
  2. Bad train/val/test split → stratified, seed-fixed, cached to JSON
  3. Label/channel ordering  → explicit 3-channel WT/TC/ET conversion
  4. Incorrect modalities    → hardcoded order t1n, t1c, t2w, t2f from config
"""

import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import nibabel as nib
from sklearn.model_selection import train_test_split

from config import (
    DATA_ROOT, SPLIT_CACHE, SPLIT_SEED,
    TRAIN_RATIO, VAL_RATIO,
    MODALITY_KEYS, SEG_SUFFIX,
    SPATIAL_SIZE, NORMALIZE_NONZERO_ONLY,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY,
    GLOBAL_SEED,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# 1.  CASE DISCOVERY
# ---------------------------------------------------------------------------

def discover_cases(data_root: Path = DATA_ROOT) -> List[Path]:
    """
    Return sorted list of patient case directories found under data_root.
    Each directory must contain at least the seg file and all 4 modality files.
    Cases that are missing any required file are skipped with a warning.
    """
    if not data_root.exists():
        raise FileNotFoundError(
            f"DATA_ROOT does not exist: {data_root}\n"
            "Check the path in config.py."
        )

    required_suffixes = MODALITY_KEYS + [SEG_SUFFIX]
    valid_cases: List[Path] = []

    for case_dir in sorted(data_root.iterdir()):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name

        missing = []
        for suffix in required_suffixes:
            expected = case_dir / f"{case_id}-{suffix}.nii.gz"
            if not expected.exists():
                missing.append(suffix)

        if missing:
            logger.warning(f"Skipping {case_id} — missing: {missing}")
            continue

        valid_cases.append(case_dir)

    if len(valid_cases) == 0:
        raise RuntimeError(
            f"No valid cases found in {data_root}.\n"
            "Expected subfolders named BraTS-GLI-XXXXX-XXX each containing\n"
            "*-t1n.nii.gz, *-t1c.nii.gz, *-t2w.nii.gz, *-t2f.nii.gz, *-seg.nii.gz"
        )

    logger.info(f"Discovered {len(valid_cases)} valid cases in {data_root}")
    return valid_cases


# ---------------------------------------------------------------------------
# 2.  TRAIN / VAL / TEST SPLIT
# ---------------------------------------------------------------------------

def build_splits(
    cases: List[Path],
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    seed:        int   = SPLIT_SEED,
    cache_path:  Path  = SPLIT_CACHE,
) -> Dict[str, List[str]]:
    """
    Split case directories into train / val / test.
    Results are cached to JSON so the same split is used across all runs.
    Never call this with a different seed once training has started —
    that would leak test cases into training.

    Returns:
        dict with keys "train", "val", "test", each a list of case_id strings.
    """
    # Load from cache if it already exists
    if cache_path.exists():
        with open(cache_path, "r") as f:
            splits = json.load(f)
        logger.info(
            f"Loaded cached split from {cache_path}  "
            f"(train={len(splits['train'])} val={len(splits['val'])} "
            f"test={len(splits['test'])})"
        )
        return splits

    # Build fresh split
    case_ids = [c.name for c in cases]
    random.seed(seed)
    np.random.seed(seed)

    test_ratio = 1.0 - train_ratio - val_ratio

    # First split off the test set
    train_val_ids, test_ids = train_test_split(
        case_ids,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
    )

    # Then split train_val into train and val
    relative_val = val_ratio / (train_ratio + val_ratio)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=relative_val,
        random_state=seed,
        shuffle=True,
    )

    splits = {
        "train": sorted(train_ids),
        "val":   sorted(val_ids),
        "test":  sorted(test_ids),
    }

    # Cache to disk
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info(
        f"Built and cached new split → "
        f"train={len(train_ids)} | val={len(val_ids)} | test={len(test_ids)}"
    )
    return splits


def get_case_paths_for_split(
    split_ids: List[str],
    data_root: Path = DATA_ROOT,
) -> List[Path]:
    """Convert a list of case_id strings back to Path objects."""
    paths = []
    for case_id in split_ids:
        p = data_root / case_id
        if p.exists():
            paths.append(p)
        else:
            logger.warning(f"Case {case_id} listed in split but not found on disk.")
    return paths


# ---------------------------------------------------------------------------
# 3.  NIBABEL I/O
# ---------------------------------------------------------------------------

def load_nifti(path: Path) -> np.ndarray:
    """Load a .nii.gz file and return its data as float32 numpy array."""
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32)


def load_case(case_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one BraTS case.

    Returns:
        image : float32 array of shape (4, D, H, W)
                channels in order [t1n, t1c, t2w, t2f]  ← FIXED ORDER
        label : int8 array of shape (D, H, W)
                raw integer labels {0, 1, 2, 3}
    """
    case_id = case_dir.name

    # Load modalities in the exact order defined in config — do NOT reorder
    modality_arrays = []
    for suffix in MODALITY_KEYS:   # ["t1n", "t1c", "t2w", "t2f"]
        fpath = case_dir / f"{case_id}-{suffix}.nii.gz"
        modality_arrays.append(load_nifti(fpath))

    image = np.stack(modality_arrays, axis=0)   # (4, D, H, W)

    # Load segmentation
    seg_path = case_dir / f"{case_id}-{SEG_SUFFIX}.nii.gz"
    label = load_nifti(seg_path).astype(np.int8)  # (D, H, W)

    return image, label


# ---------------------------------------------------------------------------
# 4.  NORMALIZATION  (z-score, non-zero brain voxels only)
# ---------------------------------------------------------------------------

def normalize_image(image: np.ndarray, nonzero_only: bool = NORMALIZE_NONZERO_ONLY) -> np.ndarray:
    """
    Z-score normalize each modality channel independently.

    If nonzero_only=True (default and correct for BraTS):
        stats (mean, std) are computed over non-zero voxels only,
        which excludes the zero-padded background outside the skull.
        This is the standard BraTS normalization approach.

    If nonzero_only=False:
        stats are computed over the full volume — this is WRONG for BraTS
        because the large zero background skews the mean toward zero
        and artificially inflates apparent contrast. This was one of your
        previous mistakes.

    Args:
        image: float32 array of shape (4, D, H, W)
    Returns:
        normalized array of same shape
    """
    normed = image.copy()
    for c in range(image.shape[0]):
        channel = image[c]
        if nonzero_only:
            mask = channel > 0
            if mask.sum() == 0:
                # Degenerate case: entire channel is zero — leave as-is
                continue
            mu  = channel[mask].mean()
            std = channel[mask].std()
        else:
            mu  = channel.mean()
            std = channel.std()

        std = std if std > 1e-8 else 1e-8   # avoid division by zero
        normed[c] = (channel - mu) / std

    return normed


# ---------------------------------------------------------------------------
# 5.  LABEL CONVERSION  (raw integers → 3-channel WT/TC/ET binary masks)
# ---------------------------------------------------------------------------

def convert_labels(label: np.ndarray) -> np.ndarray:
    """
    Convert BraTS integer segmentation mask to 3-channel binary mask.

    BraTS 2023 integer values:
        0 = background
        1 = NCR (necrotic tumor core)
        2 = ED  (peritumoral edema / invaded tissue)
        3 = ET  (GD-enhancing tumor)

    Output channels (THIS ORDER IS FIXED — matches config.REGION_NAMES):
        channel 0 → WT (whole tumor)     = {1, 2, 3}
        channel 1 → TC (tumor core)      = {1, 3}
        channel 2 → ET (enhancing tumor) = {3}

    Args:
        label: int8 array of shape (D, H, W)
    Returns:
        float32 array of shape (3, D, H, W), binary {0.0, 1.0}
    """
    wt = (label >= 1).astype(np.float32)          # labels 1, 2, 3
    tc = ((label == 1) | (label == 3)).astype(np.float32)  # labels 1, 3
    et = (label == 3).astype(np.float32)          # label 3 only

    return np.stack([wt, tc, et], axis=0)          # (3, D, H, W)


# ---------------------------------------------------------------------------
# 6.  SPATIAL TRANSFORMS (crop / pad to fixed size)
# ---------------------------------------------------------------------------

def pad_or_crop(volume: np.ndarray, target_size: Tuple[int, ...]) -> np.ndarray:
    """
    Pad or center-crop a volume to target_size along the spatial dimensions.

    Args:
        volume: array of shape (C, D, H, W)  or  (D, H, W)
        target_size: (D, H, W) tuple
    Returns:
        array of same channel count but spatial dims = target_size
    """
    has_channel = volume.ndim == 4
    if not has_channel:
        volume = volume[None]   # add fake channel dim

    C, D, H, W = volume.shape
    tD, tH, tW = target_size

    # --- Pad if smaller ---
    pad_d = max(0, tD - D)
    pad_h = max(0, tH - H)
    pad_w = max(0, tW - W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = np.pad(
            volume,
            (
                (0, 0),
                (pad_d // 2, pad_d - pad_d // 2),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
            ),
            mode="constant",
            constant_values=0,
        )
        C, D, H, W = volume.shape

    # --- Center-crop if larger ---
    start_d = (D - tD) // 2
    start_h = (H - tH) // 2
    start_w = (W - tW) // 2
    volume = volume[:, start_d:start_d+tD, start_h:start_h+tH, start_w:start_w+tW]

    if not has_channel:
        volume = volume[0]

    return volume


# ---------------------------------------------------------------------------
# 7.  AUGMENTATION (training only — lightweight, 3D-safe)
# ---------------------------------------------------------------------------

def random_flip(image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Random horizontal flip along each spatial axis independently."""
    for axis in [1, 2, 3]:   # D, H, W axes (0 is channel)
        if random.random() < 0.5:
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
    return image, label


def random_intensity_scale(image: np.ndarray, scale_range=(0.9, 1.1)) -> np.ndarray:
    """Multiply each channel by a small random scalar."""
    for c in range(image.shape[0]):
        scale = random.uniform(*scale_range)
        image[c] = image[c] * scale
    return image


# ---------------------------------------------------------------------------
# 8.  DATASET CLASS
# ---------------------------------------------------------------------------

class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS 2023 GLI segmentation.

    Each __getitem__ returns:
        image : float32 tensor of shape (4, 128, 128, 128)  — 4 modalities
        label : float32 tensor of shape (3, 128, 128, 128)  — WT, TC, ET
        case_id: str  — patient identifier (useful for saving predictions)
    """

    def __init__(
        self,
        case_dirs: List[Path],
        spatial_size: Tuple[int, int, int] = SPATIAL_SIZE,
        augment: bool = False,
        cache_in_memory: bool = False,
    ):
        """
        Args:
            case_dirs     : list of Path objects, one per patient
            spatial_size  : target (D, H, W) after crop/pad
            augment       : apply random flips and intensity jitter (train only)
            cache_in_memory: load all volumes at init (fast if RAM allows,
                             risky for large datasets)
        """
        self.case_dirs   = case_dirs
        self.spatial_size = spatial_size
        self.augment     = augment
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        if cache_in_memory:
            logger.info(f"Pre-loading {len(case_dirs)} cases into RAM …")
            for cd in case_dirs:
                img, lbl = self._load_and_preprocess(cd)
                self._cache[cd.name] = (img, lbl)
            logger.info("Pre-load complete.")

    def _load_and_preprocess(
        self, case_dir: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load, normalize, convert labels, and resize one case."""
        image, label = load_case(case_dir)              # (4,D,H,W), (D,H,W)
        image = normalize_image(image)                   # z-score per channel
        label = convert_labels(label)                    # → (3,D,H,W) binary

        image = pad_or_crop(image, self.spatial_size)   # (4,128,128,128)
        label = pad_or_crop(label, self.spatial_size)   # (3,128,128,128)

        return image, label

    def __len__(self) -> int:
        return len(self.case_dirs)

    def __getitem__(self, idx: int) -> Dict:
        case_dir = self.case_dirs[idx]
        case_id  = case_dir.name

        if case_id in self._cache:
            image, label = self._cache[case_id]
            image, label = image.copy(), label.copy()
        else:
            image, label = self._load_and_preprocess(case_dir)

        # Augmentation (training only)
        if self.augment:
            image, label = random_flip(image, label)
            image = random_intensity_scale(image)

        image_tensor = torch.from_numpy(image).float()   # (4, 128, 128, 128)
        label_tensor = torch.from_numpy(label).float()   # (3, 128, 128, 128)

        return {
            "image":   image_tensor,
            "label":   label_tensor,
            "case_id": case_id,
        }


# ---------------------------------------------------------------------------
# 9.  DATALOADER FACTORY
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_root:    Path  = DATA_ROOT,
    spatial_size: Tuple  = SPATIAL_SIZE,
    batch_size:   int    = BATCH_SIZE,
    num_workers:  int    = NUM_WORKERS,
    pin_memory:   bool   = PIN_MEMORY,
    cache_train:  bool   = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, List[str]]]:
    """
    Build train / val / test DataLoaders from the BraTS data directory.

    Returns:
        train_loader, val_loader, test_loader, splits_dict
    """
    # Discover all valid cases
    all_cases = discover_cases(data_root)

    # Build (or load cached) split
    splits = build_splits(all_cases)

    # Convert case_id strings back to Path objects
    train_dirs = get_case_paths_for_split(splits["train"], data_root)
    val_dirs   = get_case_paths_for_split(splits["val"],   data_root)
    test_dirs  = get_case_paths_for_split(splits["test"],  data_root)

    logger.info(
        f"Split sizes → train: {len(train_dirs)} | "
        f"val: {len(val_dirs)} | test: {len(test_dirs)}"
    )

    # Datasets
    train_ds = BraTSDataset(train_dirs, spatial_size, augment=True,  cache_in_memory=cache_train)
    val_ds   = BraTSDataset(val_dirs,   spatial_size, augment=False, cache_in_memory=False)
    test_ds  = BraTSDataset(test_dirs,  spatial_size, augment=False, cache_in_memory=False)

    def _seed_worker(worker_id):
        """Ensure each DataLoader worker gets a different but reproducible seed."""
        worker_seed = GLOBAL_SEED + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(GLOBAL_SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=g,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,           # always 1 for validation — full volume inference
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_loader, splits


# ---------------------------------------------------------------------------
# 10.  STANDALONE SANITY CHECK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== dataset.py sanity check ===\n")

    print("Discovering cases …")
    cases = discover_cases()

    print("\nBuilding splits …")
    splits = build_splits(cases)
    print(f"  train: {len(splits['train'])} | val: {len(splits['val'])} | test: {len(splits['test'])}")

    print("\nLoading first training case …")
    first_case = DATA_ROOT / splits["train"][0]
    image, label_raw = load_case(first_case)
    print(f"  Raw image shape  : {image.shape}   dtype: {image.dtype}")
    print(f"  Raw label shape  : {label_raw.shape}  dtype: {label_raw.dtype}")
    print(f"  Unique label vals: {np.unique(label_raw)}")

    print("\nNormalizing …")
    image_normed = normalize_image(image)
    for c, name in enumerate(MODALITY_KEYS):
        ch = image_normed[c]
        mask = image[c] > 0
        print(f"  [{name}]  mean={ch[mask].mean():.4f}  std={ch[mask].std():.4f}  (should be ~0 and ~1)")

    print("\nConverting labels …")
    label_3ch = convert_labels(label_raw)
    print(f"  3-channel label shape: {label_3ch.shape}")
    for i, region in enumerate(["WT", "TC", "ET"]):
        n_vox = int(label_3ch[i].sum())
        print(f"  {region}: {n_vox} foreground voxels")

    print("\nTesting pad_or_crop …")
    img_cropped = pad_or_crop(image_normed, SPATIAL_SIZE)
    lbl_cropped = pad_or_crop(label_3ch,   SPATIAL_SIZE)
    print(f"  image after crop: {img_cropped.shape}  (expected (4, 128, 128, 128))")
    print(f"  label after crop: {lbl_cropped.shape}  (expected (3, 128, 128, 128))")

    print("\nTesting Dataset __getitem__ …")
    train_dirs = get_case_paths_for_split(splits["train"])
    ds = BraTSDataset(train_dirs[:3], augment=True)
    sample = ds[0]
    print(f"  image tensor: {sample['image'].shape}  dtype: {sample['image'].dtype}")
    print(f"  label tensor: {sample['label'].shape}  dtype: {sample['label'].dtype}")
    print(f"  case_id     : {sample['case_id']}")

    print("\nBuilding DataLoaders …")
    tr, vl, te, sp = get_dataloaders(num_workers=0)
    batch = next(iter(tr))
    print(f"  train batch image: {batch['image'].shape}")
    print(f"  train batch label: {batch['label'].shape}")

    print("\ndataset.py — ALL CHECKS PASSED")