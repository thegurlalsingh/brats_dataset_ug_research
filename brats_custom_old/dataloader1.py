import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import nibabel as nib


class BraTSDataset(Dataset):
    def __init__(self, root_dir, k=3, transform=None, crop=True):
        """
        2.5D BraTS Dataset.
        FIXES:
          1. Volume-level in-memory cache (_vol_cache) — each patient loaded once per process,
             not once per __getitem__ call. Eliminates the dominant cause of slow training.
          2. BraTS 2023 label mapping: ET = label 3 (not 4), TC = {1, 3}, WT = {1, 2, 3}.
        """
        self.root_dir = root_dir
        self.k = k
        self.pad = k // 2
        self.transform = transform
        self.crop = crop

        # Robust patient filtering
        all_items = os.listdir(root_dir)
        self.patients = sorted([
            p for p in all_items
            if os.path.isdir(os.path.join(root_dir, p)) and not p.startswith('.')
        ])

        print(f"Found {len(self.patients)} valid patient folders")

        self.data = []  # list of (patient_path, central_slice_z)

        for p in self.patients:
            p_path = os.path.join(root_dir, p)
            seg_path = None
            for f in os.listdir(p_path):
                if f.endswith("-seg.nii.gz"):
                    seg_path = os.path.join(p_path, f)
                    break

            if seg_path is None:
                print(f"Warning: Missing seg.nii.gz in {p} → skipping")
                continue

            try:
                seg = nib.load(seg_path).get_fdata()
                D = seg.shape[2]
                num_samples = int(0.3 * D)
                selected_slices = np.linspace(0, D - 1, num_samples, dtype=int)
                for z in selected_slices:
                    self.data.append((p_path, int(z)))
            except Exception as e:
                print(f"Error loading {p}: {e} → skipping")

        # FIX 1: Volume-level cache — keyed by patient path
        # Stores (list_of_4_modality_vols, seg_vol, crop_slices)
        # Populated lazily on first access, then reused for all slices of that patient.
        self._vol_cache = {}

    def __len__(self):
        return len(self.data)

    def _normalize(self, vol):
        mask = vol > 0.01
        if np.sum(mask) == 0:
            return vol.astype(np.float32)
        mean = vol[mask].mean()
        std = vol[mask].std()
        return ((vol - mean) / (std + 1e-8)).astype(np.float32)

    def _get_safe_crop(self, modalities, seg, pad=25):
        brain_mask = np.zeros_like(seg, dtype=bool)
        for mod in modalities:
            brain_mask |= (mod > 0.01)
        brain_mask |= (seg > 0)

        if not np.any(brain_mask):
            return None

        coords = np.argwhere(brain_mask)
        minx, maxx = coords[:, 0].min(), coords[:, 0].max()
        miny, maxy = coords[:, 1].min(), coords[:, 1].max()
        minz, maxz = coords[:, 2].min(), coords[:, 2].max()

        minx = max(0, minx - pad)
        maxx = min(seg.shape[0], maxx + pad)
        miny = max(0, miny - pad)
        maxy = min(seg.shape[1], maxy + pad)
        minz = max(0, minz - pad)
        maxz = min(seg.shape[2], maxz + pad)

        return (slice(minx, maxx), slice(miny, maxy), slice(minz, maxz))

    def _load_patient(self, p_path):
        """Load + normalize all modalities. Called once per patient thanks to cache."""
        modalities_paths = {
            "t1n": None, "t1c": None,
            "t2w": None, "t2f": None, "seg": None
        }

        for f in os.listdir(p_path):
            fpath = os.path.join(p_path, f)
            if f.endswith("-t1n.nii.gz"):
                modalities_paths["t1n"] = fpath
            elif f.endswith("-t1c.nii.gz"):
                modalities_paths["t1c"] = fpath
            elif f.endswith("-t2w.nii.gz"):
                modalities_paths["t2w"] = fpath
            elif f.endswith("-t2f.nii.gz"):
                modalities_paths["t2f"] = fpath
            elif f.endswith("-seg.nii.gz"):
                modalities_paths["seg"] = fpath

        for k, v in modalities_paths.items():
            if v is None:
                raise FileNotFoundError(f"Missing modality {k} in {p_path}")

        vols = []
        for key in ["t1n", "t1c", "t2w", "t2f"]:
            vol = nib.load(modalities_paths[key]).get_fdata()
            vol = self._normalize(vol)
            vols.append(vol)

        seg = nib.load(modalities_paths["seg"]).get_fdata()
        return vols, seg

    def _get_patient_data(self, p_path):
        """
        Returns (cropped_modalities, cropped_seg, crop_slices) from cache.
        Loads from disk only on first call for this patient.
        """
        if p_path not in self._vol_cache:
            vols, seg = self._load_patient(p_path)

            crop_slices = None
            if self.crop:
                crop_slices = self._get_safe_crop(vols, seg, pad=25)

            if crop_slices is not None:
                vols = [m[crop_slices] for m in vols]
                seg = seg[crop_slices]

            # Store as float32 numpy arrays to keep memory reasonable
            self._vol_cache[p_path] = (
                [v.astype(np.float32) for v in vols],
                seg.astype(np.float32),
                crop_slices
            )

        return self._vol_cache[p_path]

    def __getitem__(self, idx):
        p_path, z = self.data[idx]

        # FIX 1: Get from cache — no disk I/O after first access
        modalities, seg, crop_slices = self._get_patient_data(p_path)

        D = seg.shape[2]

        # Adjust z index after cropping
        if crop_slices is not None:
            z = z - crop_slices[2].start
            z = int(np.clip(z, 0, D - 1))

        # Stack k adjacent slices → (4*k, H, W)
        slice_stack = []
        for i in range(z - self.pad, z + self.pad + 1):
            i_clipped = int(np.clip(i, 0, D - 1))
            for mod in modalities:
                slice_stack.append(mod[:, :, i_clipped])

        target_size = (192, 192)

        x = np.stack(slice_stack, axis=0).astype(np.float32)
        x = torch.from_numpy(x).unsqueeze(0)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = x.squeeze(0)

        # FIX 2: Correct BraTS 2023 label mapping
        # Old (BraTS 2021): ET=4, TC={1,4}, WT={1,2,4}
        # New (BraTS 2023): ET=3, TC={1,3}, WT={1,2,3}
        y_slice = seg[:, :, z]

        y_wt = np.isin(y_slice, [1, 2, 3]).astype(np.float32)   # Whole Tumour
        y_tc = np.isin(y_slice, [1, 3]).astype(np.float32)       # Tumour Core
        y_et = (y_slice == 3).astype(np.float32)                 # Enhancing Tumour

        y = np.stack([y_wt, y_tc, y_et], axis=0).astype(np.float32)
        y = torch.from_numpy(y).unsqueeze(0)
        y = F.interpolate(y, size=target_size, mode='nearest')
        y = y.squeeze(0)

        if self.transform:
            x, y = self.transform(x, y)

        return x, y