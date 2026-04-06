# =====================================================
# FINAL: nnU-Net + Forward CAM + Mask XAI + Evaluation
# =====================================================

import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import pandas as pd

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# ================= PATHS =================
os.environ['nnUNet_raw'] = r"C:\Users\Gurlal-Stu\Downloads\brats\output_training\nnUNet_raw"
os.environ['nnUNet_preprocessed'] = r"C:\Users\Gurlal-Stu\Downloads\brats\output_training\nnUNet_preprocessed"
os.environ['nnUNet_results'] = r"C:\Users\Gurlal-Stu\Downloads\brats\output_training\nnUNet_results"

TEST_FOLDER = Path(r"C:\Users\Gurlal-Stu\Downloads\brats\test")
MODEL_FOLDER = Path(
    r"C:\Users\Gurlal-Stu\Downloads\brats\output_training\nnUNet_results\Dataset001_BraTS2023\nnUNetTrainer__nnUNetPlans__3d_fullres"
)
SAVE_DIR = Path(r"./FINAL_XAI_OUTPUT")

SAVE_DIR.mkdir(exist_ok=True)
(SAVE_DIR / "figures").mkdir(exist_ok=True)
(SAVE_DIR / "npy").mkdir(exist_ok=True)
(SAVE_DIR / "preds").mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# MODEL
# =====================================================
def load_model():
    predictor = nnUNetPredictor(device=DEVICE, verbose=False)

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(MODEL_FOLDER),
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth'
    )

    return predictor, predictor.network.eval()

# =====================================================
# FORWARD CAM
# =====================================================
class ForwardCAM:
    def __init__(self, model):
        self.activations = None
        layer = dict(model.named_modules())['decoder.seg_layers.4']
        layer.register_forward_hook(self.hook)

    def hook(self, module, inp, out):
        self.activations = out.detach()

    def get(self):
        cam = torch.norm(self.activations, p=2, dim=1)
        cam = cam[0].cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# =====================================================
# MASK XAI
# =====================================================
def mask_xai(predictor, case_files, patch=16):

    data_iter = predictor._internal_get_data_iterator_from_lists_of_filenames(
        [case_files], None, None, 1
    )
    data = next(data_iter)['data'].to(DEVICE)

    with torch.no_grad():
        base_logits = predictor.predict_logits_from_preprocessed_data(data)

    # ✅ use probabilities
    base_probs = torch.softmax(base_logits, dim=1)
    base_score = base_probs[:, 3].mean().item()

    C, D, H, W = data.shape
    importance = np.zeros((D, H, W))
    counts = np.zeros((D, H, W))   # ✅ NEW

    stride = patch // 2  # ✅ overlap

    for z in range(0, D, stride):
        for y in range(0, H, stride):
            for x in range(0, W, stride):

                masked = data.clone()

                z2 = min(z+patch, D)
                y2 = min(y+patch, H)
                x2 = min(x+patch, W)

                region = masked[:, z:z2, y:y2, x:x2]
                mean_val = region.mean()

                masked[:, z:z2, y:y2, x:x2] = mean_val

                with torch.no_grad():
                    logits = predictor.predict_logits_from_preprocessed_data(masked)

                probs = torch.softmax(logits, dim=1)   # ✅ FIX
                drop = base_score - probs[:, 3].mean().item()

                importance[z:z2, y:y2, x:x2] += drop   # ✅ accumulate
                counts[z:z2, y:y2, x:x2] += 1

    importance = importance / (counts + 1e-8)

    importance = gaussian_filter(importance, sigma=2)
    importance = np.clip(importance, 0, None)

    # ✅ FIX normalization
    return (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)

# =====================================================
# UTILS
# =====================================================
def resize_cam(cam, target_shape):
    cam_t = torch.tensor(cam).unsqueeze(0).unsqueeze(0).float()
    cam_resized = F.interpolate(cam_t, size=target_shape, mode='trilinear', align_corners=False)
    return cam_resized[0, 0].cpu().numpy()

def dice_binary(p, g):
    inter = (p & g).sum()
    return 2 * inter / (p.sum() + g.sum() + 1e-8)

def modality_focus(xai, t1c, t2f, gt):

    tumor = (gt > 0)

    x = xai[tumor]
    t1 = t1c[tumor]
    t2 = t2f[tumor]

    if len(x) < 10:
        return {"T1c_corr": 0.0, "T2f_corr": 0.0}

    return {
        "T1c_corr": float(np.corrcoef(x, t1)[0,1]),
        "T2f_corr": float(np.corrcoef(x, t2)[0,1])
    }

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":

    predictor, model = load_model()
    results = []

    for patient in tqdm(sorted(TEST_FOLDER.glob("BraTS-*"))):

        pid = patient.name
        print("\nProcessing:", pid)

        case_files = [
            str(patient / f"{pid}-t1n.nii.gz"),
            str(patient / f"{pid}-t1c.nii.gz"),
            str(patient / f"{pid}-t2w.nii.gz"),
            str(patient / f"{pid}-t2f.nii.gz")
        ]

        raw = np.stack([nib.load(f).get_fdata() for f in case_files])
        gt = nib.load(patient / f"{pid}-seg.nii.gz").get_fdata()

        # ---- Prediction ----
        out_file = SAVE_DIR / "preds" / f"{pid}.nii.gz"

        predictor.predict_from_files(
            [case_files],
            [str(out_file)],
            save_probabilities=False
        )

        pred = nib.load(out_file).get_fdata()

        # ---- XAI ----
        cam_extractor = ForwardCAM(model)

        data_iter = predictor._internal_get_data_iterator_from_lists_of_filenames(
            [case_files], None, None, 1
        )
        data = next(data_iter)['data'].to(DEVICE)

        with torch.no_grad():
            _ = predictor.predict_logits_from_preprocessed_data(data)

        forward_cam = cam_extractor.get()
        mask_map = mask_xai(predictor, case_files)

        # ---- Resize ----
        forward_cam_r = resize_cam(forward_cam, raw.shape[1:])
        mask_map_r = resize_cam(mask_map, raw.shape[1:])

        # ---- Save ----
        np.save(SAVE_DIR / "npy" / f"{pid}_forward.npy", forward_cam_r)
        np.save(SAVE_DIR / "npy" / f"{pid}_mask.npy", mask_map_r)

        # ---- Metrics ----
        dice_et = dice_binary(pred == 3, gt == 3)
        dice_wt = dice_binary(pred > 0, gt > 0)

        focus = modality_focus(mask_map_r, raw[1], raw[3], gt)

        result = {
            "patient": pid,
            "dice_et": float(dice_et),
            "dice_wt": float(dice_wt),
            "t1c_focus": focus["T1c_corr"],
            "t2f_focus": focus["T2f_corr"]
        }

        results.append(result)
        print(result)

        # ---- Visualization ----
        mid = mask_map_r.shape[2] // 2
        t1c = raw[1, :, :, mid]

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.imshow(t1c, cmap='gray')
        plt.imshow(forward_cam_r[:,:,mid], alpha=0.5, cmap='jet')
        plt.title("Forward CAM")

        plt.subplot(1,2,2)
        plt.imshow(t1c, cmap='gray')
        plt.imshow(mask_map_r[:,:,mid], alpha=0.5, cmap='jet')
        plt.title("Mask XAI")

        plt.savefig(SAVE_DIR / "figures" / f"{pid}.png")
        plt.close()

    pd.DataFrame(results).to_csv(SAVE_DIR / "metrics.csv", index=False)

    print("\n✅ DONE — NO ERRORS")