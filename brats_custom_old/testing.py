import torch
import numpy as np
import nibabel as nib
import os
import torch.nn.functional as F

# ============================
# 🔹 CONFIG
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"C:\Users\Gurlal-Stu\Downloads\custom\scripts\checkpoints\best_model.pth"   # upload this
PATIENT_PATH = r"C:\Users\Gurlal-Stu\Downloads\custom\data\BraTS-GLI-00250-000"  # upload 1 patient folder
K = 3
TARGET_SIZE = (192, 192)

# ============================
# 🔹 LOAD MODEL
# ============================
from preprocess2 import Hybrid2_5DBackbone
from sparse_selection4 import SparseSelectionModule
from volumetric_context5 import DynamicSparseRefinerModel
from fusion_layer6 import FusionLayer

class FullModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Hybrid2_5DBackbone(k=3, base_channels=64)
        self.sparse = SparseSelectionModule(64, patch_size=8, num_slices=3)
        self.refiner3d = DynamicSparseRefinerModel(self.sparse)
        self.fusion = FusionLayer(64, 3)

        self.coarse_head = torch.nn.Conv2d(64, 3, 1)
        self.imp_head = torch.nn.Conv2d(64, 1, 1)

    def forward(self, x):
        features = self.backbone(x)

        coarse = self.coarse_head(features)
        importance = torch.sigmoid(self.imp_head(features))

        refiner_out = self.refiner3d(x)
        refined_3d = refiner_out["feat3d"]
        refined_2d = refined_3d[:, :, refined_3d.shape[2]//2]

        unc = torch.sigmoid(coarse) * (1 - torch.sigmoid(coarse))
        unc, _ = torch.max(unc, dim=1, keepdim=True)

        final = self.fusion(features, coarse, importance, unc, refined_2d)

        return final

model = FullModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model Loaded!")

# ============================
# 🔹 LOAD PATIENT
# ============================
def load_modality(path):
    return nib.load(path).get_fdata()

def normalize(vol):
    mask = vol > 0
    if mask.sum() == 0:
        return vol
    return (vol - vol[mask].mean()) / (vol[mask].std() + 1e-8)

def load_patient(p_path):
    files = os.listdir(p_path)

    t1n = [f for f in files if "t1n" in f][0]
    t1c = [f for f in files if "t1c" in f][0]
    t2w = [f for f in files if "t2w" in f][0]
    t2f = [f for f in files if "t2f" in f][0]
    seg = [f for f in files if "seg" in f][0]

    vols = [
        normalize(load_modality(os.path.join(p_path, t1n))),
        normalize(load_modality(os.path.join(p_path, t1c))),
        normalize(load_modality(os.path.join(p_path, t2w))),
        normalize(load_modality(os.path.join(p_path, t2f)))
    ]

    seg = load_modality(os.path.join(p_path, seg))

    return vols, seg

modalities, seg = load_patient(PATIENT_PATH)

# ============================
# 🔹 INFERENCE (SLICE LOOP)
# ============================
pad = K // 2
D = seg.shape[2]

pred_volume = []

with torch.no_grad():
    for z in range(D):

        slice_stack = []
        for i in range(z - pad, z + pad + 1):
            i = np.clip(i, 0, D - 1)
            for mod in modalities:
                slice_stack.append(mod[:, :, i])

        x = np.stack(slice_stack, axis=0).astype(np.float32)
        x = torch.from_numpy(x).unsqueeze(0).to(DEVICE)

        x = F.interpolate(x, size=TARGET_SIZE, mode='bilinear', align_corners=False)

        pred = model(x)
        pred = torch.sigmoid(pred).cpu().numpy()[0]

        pred_volume.append(pred)

pred_volume = np.stack(pred_volume, axis=-1)  # (3, H, W, D)

# ============================
# 🔹 GROUND TRUTH
# ============================
y_wt = (seg > 0).astype(np.float32)
y_tc = np.isin(seg, [1, 4]).astype(np.float32)
y_et = (seg == 4).astype(np.float32)

# resize GT
gt = np.stack([y_wt, y_tc, y_et], axis=0)
gt_resized = np.zeros_like(pred_volume)

for i in range(3):
    for z in range(D):
        gt_resized[i, :, :, z] = F.interpolate(
            torch.tensor(gt[i, :, :, z]).unsqueeze(0).unsqueeze(0),
            size=TARGET_SIZE,
            mode='nearest'
        ).squeeze().numpy()

# ============================
# 🔹 DICE FUNCTION
# ============================
def dice(pred, gt, thresh=0.5):
    pred = (pred > thresh).astype(np.float32)

    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum()

    return (2 * intersection + 1e-5) / (union + 1e-5)

# ============================
# 🔥 FINAL METRICS
# ============================
wt_dice = dice(pred_volume[0], gt_resized[0])
tc_dice = dice(pred_volume[1], gt_resized[1])
et_dice = dice(pred_volume[2], gt_resized[2])

print("\n🔥 Patient-wise Dice Scores:")
print(f"WT Dice : {wt_dice:.4f}")
print(f"TC Dice : {tc_dice:.4f}")
print(f"ET Dice : {et_dice:.4f}")
print("GT ET sum:", gt_resized[2].sum())
print("Pred ET sum:", pred_volume[2].sum())