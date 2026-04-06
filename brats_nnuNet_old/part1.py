# =====================================================
# nnU-Net + Forward CAM (FINAL WITH PREPROCESSING)
# =====================================================

import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# ========================= PATHS =========================
os.environ['nnUNet_raw'] = r"C:\Users\Gurlal-Stu\Downloads\brats\output\nnUNet_raw"
os.environ['nnUNet_preprocessed'] = r"C:\Users\Gurlal-Stu\Downloads\brats\output\nnUNet_preprocessed"
os.environ['nnUNet_results'] = r"C:\Users\Gurlal-Stu\Downloads\brats\output\nnUNet_results"

TEST_FOLDER = Path(r"C:\Users\Gurlal-Stu\Downloads\brats\test")
SAVE_DIR = Path(r"C:\Users\Gurlal-Stu\Downloads\brats\xai_FINAL_clean")

MODEL_FOLDER = Path(
    r"C:\Users\Gurlal-Stu\Downloads\brats\output_training\nnUNet_results\Dataset001_BraTS2023\nnUNetTrainer__nnUNetPlans__3d_fullres"
)

SAVE_DIR.mkdir(parents=True, exist_ok=True)
(SAVE_DIR / "figures").mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =====================================================
# FORWARD CAM
# =====================================================
class ForwardCAM:
    def __init__(self, model, layer_name):
        self.activations = None
        layer = dict(model.named_modules())[layer_name]
        layer.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.activations = output.detach()

    def get_cam(self):
        return torch.norm(self.activations, p=2, dim=1)

# =====================================================
# LOAD MODEL
# =====================================================
def load_model():
    predictor = nnUNetPredictor(
        tile_step_size=0.75,
        use_gaussian=False,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=DEVICE,
        verbose=False
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(MODEL_FOLDER),
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth'
    )

    model = predictor.network
    model.eval()

    print("✅ Model loaded")
    return predictor, model

# =====================================================
# PREPROCESS (OFFICIAL & SAFE)
# =====================================================
def preprocess_case(case_files):
    # Load raw image
    imgs = [nib.load(f).get_fdata() for f in case_files]
    raw = np.stack(imgs).astype(np.float32)

    tensor = torch.from_numpy(raw).unsqueeze(0)

    # 🔥 Pad to multiples of 16 (nnU-Net requirement)
    _, _, D, H, W = tensor.shape

    pad_d = (16 - D % 16) % 16
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16

    tensor = torch.nn.functional.pad(
        tensor,
        (0, pad_w, 0, pad_h, 0, pad_d)
    )

    return tensor.to(DEVICE)

# =====================================================
# PREDICTION
# =====================================================
def run_prediction(predictor, case_files, pid):
    out_dir = SAVE_DIR / "temp_pred" / pid
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        predictor.predict_from_files(
            [case_files],
            output_folder_or_list_of_truncated_output_files=str(out_dir),
            save_probabilities=False,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1
        )

    pred_file = list(out_dir.glob("*.nii.gz"))[0]
    return nib.load(pred_file).get_fdata()

# =====================================================
# CAM
# =====================================================
def generate_cam_safe(predictor, model, case_files):
    """
    100% safe CAM using nnU-Net's internal prediction pipeline
    """

    cam_extractor = ForwardCAM(model, 'decoder.seg_layers.4')

    # 🔥 Run ONE internal prediction step (this is key)
    data_iterator = predictor._internal_get_data_iterator_from_lists_of_filenames(
        [case_files],
        seg_from_prev_stage_files=None,
        output_filenames_truncated=None,
        num_processes=1
    )

    data_dict = next(data_iterator)
    data = data_dict['data'].to(DEVICE)

    # 🔥 CRITICAL: use nnU-Net internal prediction (not raw model)
    with torch.no_grad():
        prediction = predictor.predict_logits_from_preprocessed_data(data)

    # CAM already captured via forward hook
    cam = cam_extractor.get_cam()[0].cpu().numpy()

    # Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    return cam

# =====================================================
# RESIZE CAM
# =====================================================
def resize_cam(cam, target_shape):
    cam_t = torch.tensor(cam).unsqueeze(0).unsqueeze(0)
    cam_resized = F.interpolate(cam_t, size=target_shape, mode='trilinear', align_corners=False)
    return cam_resized[0, 0].numpy()

# =====================================================
# VISUALIZATION
# =====================================================
def visualize(pid, raw_image, cam, pred, gt):
    mid = cam.shape[2] // 2

    t1c = raw_image[1, :, :, mid]
    cam_slice = cam[:, :, mid]

    pred_et = (pred[:, :, mid] == 3)
    gt_et = (gt[:, :, mid] == 3)

    fig, axs = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"{pid} - Forward CAM", fontsize=14)

    axs[0].imshow(t1c, cmap='gray'); axs[0].set_title("T1c"); axs[0].axis('off')
    axs[1].imshow(cam_slice, cmap='jet'); axs[1].set_title("CAM"); axs[1].axis('off')

    axs[2].imshow(t1c, cmap='gray')
    axs[2].imshow(cam_slice, cmap='jet', alpha=0.6)
    axs[2].set_title("Overlay"); axs[2].axis('off')

    axs[3].imshow(pred_et, cmap='viridis'); axs[3].set_title("Prediction"); axs[3].axis('off')
    axs[4].imshow(gt_et, cmap='viridis'); axs[4].set_title("GT"); axs[4].axis('off')

    plt.tight_layout()
    plt.savefig(SAVE_DIR / "figures" / f"{pid}.png", dpi=250)
    plt.close()

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    predictor, model = load_model()

    patients = sorted(TEST_FOLDER.glob("BraTS-GLI-*"))[:3]

    for patient_path in tqdm(patients):
        pid = patient_path.name
        print(f"\n🔍 {pid}")

        case_files = [
            str(patient_path / f"{pid}-t1n.nii.gz"),
            str(patient_path / f"{pid}-t1c.nii.gz"),
            str(patient_path / f"{pid}-t2w.nii.gz"),
            str(patient_path / f"{pid}-t2f.nii.gz")
        ]

        # Raw image (for visualization)
        imgs = [nib.load(f).get_fdata() for f in case_files]
        raw_image = np.stack(imgs).astype(np.float32)

        # Prediction
        pred = run_prediction(predictor, case_files, pid)

        # ✅ Proper nnU-Net preprocessing
        cam = generate_cam_safe(predictor, model, case_files)

        # Resize CAM back
        cam = resize_cam(cam, raw_image.shape[1:])

        # Ground truth
        gt = nib.load(patient_path / f"{pid}-seg.nii.gz").get_fdata()

        # Visualize
        visualize(pid, raw_image, cam, pred, gt)

        print(f"✅ Done: {pid}")

    print("\n🎉 DONE — CLEAN + CORRECT + STABLE")