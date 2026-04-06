# =============================================================================
#   nnU-Net v2 XAI SCRIPT for Test Dataset (Only 3 Patients)
#   Custom Grad-CAM + SHAP + LIME + Truthfulness + Faithfulness + Bounding Boxes
# =============================================================================

import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm.auto import tqdm
import pandas as pd
from captum.attr import GradientShap
from sklearn.metrics import auc
import torch.nn.functional as F
from pathlib import Path
from lime import lime_image
from skimage.segmentation import mark_boundaries

# ========================= CONFIGURATION =========================
MODEL_FOLDER = r"C:\Users\Gurlal-Stu\Downloads\brats\output_training\nnUNet_results\Dataset001_BraTS2023\nnUNetTrainer__nnUNetPlans__3d_fullres"
TEST_DATA_ROOT = r"C:\Users\Gurlal-Stu\Downloads\brats\test"   # ← Your folder with 3 patients
SAVE_DIR = r"C:\Users\Gurlal-Stu\Downloads\brats\output_xAI_test"
CHECKPOINT_PATH = r"C:\Users\Gurlal-Stu\Downloads\brats\output_training\nnUNet_results\Dataset001_BraTS2023\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_0\checkpoint_best.pth"

N_CASES_TO_PROCESS = 3                                          # Only 3 patients
SLICE_AXIS = 2
MID_SLICE_OFFSET = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHAP_NSAMPLES = 25
LIME_N_SAMPLES = 600

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "shap"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "lime"), exist_ok=True)

# ========================= LOAD nnU-Net v2 MODEL =========================
print("Loading nnU-Net model using Predictor...")

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.paths import nnUNet_preprocessed

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=DEVICE,
    verbose=False,
    allow_tqdm=True
)

predictor.initialize_from_trained_model_folder(
    model_training_output_dir=MODEL_FOLDER,
    use_folds=(0,),
    checkpoint_name='checkpoint_best.pth'
)

model = predictor.network
model.eval()
model.to(DEVICE)
preprocessor = DefaultPreprocessor()
print("✅ Model loaded successfully!")
print("Model type:", type(model))

# ========================= Custom 3D Grad-CAM =========================
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class=3):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, target_class].mean()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='trilinear', align_corners=False)
        cam = cam[0, 0].cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

target_layer = None
for name, module in reversed(list(model.named_modules())):
    if isinstance(module, torch.nn.Conv3d):
        target_layer = module
        print(f"Using target layer: {name}")
        break

grad_cam = GradCAM3D(model, target_layer)

# ========================= HELPERS =========================
def load_case(patient_folder: Path):
    p = patient_folder
    data = {
        't1n': nib.load(p / f"{p.name}-t1n.nii").get_fdata(),
        't1c': nib.load(p / f"{p.name}-t1c.nii").get_fdata(),
        't2w': nib.load(p / f"{p.name}-t2w.nii").get_fdata(),
        't2f': nib.load(p / f"{p.name}-t2f.nii").get_fdata(),
        'seg': nib.load(p / f"{p.name}-seg.nii").get_fdata(),
    }
    img = np.stack([data[k] for k in ['t1n', 't1c', 't2w', 't2f']], axis=0)
    return img, data['seg']

def preprocess_input(img_np):
    return torch.from_numpy(img_np).float().to(DEVICE)[None, ...]

def get_gt_regions(seg):
    return {
        'ET': (seg == 3).astype(np.float32),
        'TC': ((seg == 1) | (seg == 3)).astype(np.float32),
        'WT': ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.float32)
    }

def dice_score(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return 2 * inter / (union + inter + 1e-8)

def get_bounding_box(mask_2d):
    if mask_2d.sum() == 0:
        return None
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return ymin, ymax, xmin, xmax

# ========================= MAIN LOOP (Only 3 patients) =========================
patients = sorted(Path(TEST_DATA_ROOT).glob("BraTS-GLI-*"))[:N_CASES_TO_PROCESS]

truth_results = []
faith_results = []
shap_results = []

for patient_path in tqdm(patients, desc="XAI on Test Dataset"):
    pid = patient_path.name
    print(f"\n🔍 Processing test patient: {pid}")

    img_np, seg_np = load_case(patient_path)
    # Use nnU-Net's preprocessor to get correct shape
    # This is the key fix
    data_dict = {
        'data': img_np.astype(np.float32),
        'seg': seg_np[None].astype(np.float32),
        'properties': {'spacing': [1.0, 1.0, 1.0]}   # assume isotropic
    }

    # Preprocess using nnU-Net's own logic
    preprocessed, _ = preprocessor.run(data_dict, model.plans_manager, model.configuration_manager)

    input_tensor = torch.from_numpy(preprocessed['data']).float().to(DEVICE)[None, ...]

    with torch.no_grad():
        pred_logits = model(input_tensor)
    pred_softmax = F.softmax(pred_logits, dim=1)
    pred_seg = pred_softmax.argmax(dim=1)[0].cpu().numpy()

    gt_regions = get_gt_regions(seg_np)

    # ====================== GRAD-CAM + TRUTHFULNESS + BOUNDING BOX ======================
    fig, axes = plt.subplots(3, 5, figsize=(22, 12), dpi=160)
    fig.suptitle(f"Test Patient {pid} - Grad-CAM | Truthfulness | Bounding Box", fontsize=15)

    for row, region in enumerate(['ET', 'TC', 'WT']):
        attr_np = grad_cam(input_tensor, target_class=3)

        mid_slice = attr_np.shape[SLICE_AXIS] // 2 + MID_SLICE_OFFSET
        bg_key = 't1c' if region == 'ET' else 't2f'
        bg_idx = ['t1n','t1c','t2w','t2f'].index(bg_key)
        bg_slice = img_np[bg_idx, ..., mid_slice]

        axes[row, 0].imshow(bg_slice, cmap='gray')
        axes[row, 0].set_title(f"{region} Original")
        axes[row, 0].axis('off')

        im = axes[row, 1].imshow(attr_np[..., mid_slice], cmap='jet', alpha=0.75)
        axes[row, 1].imshow(bg_slice, cmap='gray', alpha=0.4)
        axes[row, 1].set_title("Grad-CAM")
        axes[row, 1].axis('off')
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046)

        # Truthfulness
        thresh = np.percentile(attr_np, 80)
        binary_heatmap = (attr_np > thresh).astype(np.float32)
        gt_mask = gt_regions[region][..., mid_slice]

        dice_val = dice_score(binary_heatmap, gt_mask)
        prec = np.logical_and(binary_heatmap, gt_mask).sum() / (binary_heatmap.sum() + 1e-8)
        rec = np.logical_and(binary_heatmap, gt_mask).sum() / (gt_mask.sum() + 1e-8)

        truth_results.append({"patient": pid, "region": region, "dice": dice_val, "precision": prec, "recall": rec})

        axes[row, 2].imshow(binary_heatmap, cmap='gray')
        axes[row, 2].set_title(f"Expl Map (Dice={dice_val:.3f})")
        axes[row, 2].axis('off')

        # Bounding Box
        ax_bb = axes[row, 3]
        ax_bb.imshow(bg_slice, cmap='gray')
        ax_bb.set_title("GT (Green) vs Pred (Red)")
        ax_bb.axis('off')

        gt_box = get_bounding_box(gt_mask)
        if gt_box:
            ymin, ymax, xmin, xmax = gt_box
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2.5, edgecolor='lime', facecolor='none', label='GT')
            ax_bb.add_patch(rect)

        pred_mask = (pred_seg[..., mid_slice] > 0).astype(np.float32)
        pred_box = get_bounding_box(pred_mask)
        if pred_box:
            ymin, ymax, xmin, xmax = pred_box
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2.5, edgecolor='red', facecolor='none', label='Predicted')
            ax_bb.add_patch(rect)

        if gt_box or pred_box:
            ax_bb.legend(loc='upper right')

        axes[row, 4].imshow(gt_mask, cmap='gray')
        axes[row, 4].set_title("Ground Truth")
        axes[row, 4].axis('off')

    fig.savefig(os.path.join(SAVE_DIR, "figures", f"{pid}_test_analysis.png"), bbox_inches='tight', dpi=180)
    plt.close(fig)

        # ====================== LIME ======================
    axial_idx = img_np.shape[-1] // 2
    img_slice = img_np[..., axial_idx].transpose(1, 2, 0)
    img_slice_uint8 = (img_slice * 255).astype(np.uint8)[:, :, :3]

    def classifier_fn(images):
        batch = torch.from_numpy(images.transpose(0, 3, 1, 2)).float().to(DEVICE) / 255.0
        batch = F.interpolate(batch, size=(img_np.shape[2], img_np.shape[3]), mode='bilinear')
        batch_full = torch.cat([batch, batch[:, :1]], dim=1)
        with torch.no_grad():
            out = model(batch_full.unsqueeze(2))
            out = out.squeeze(2)
        return F.softmax(out.mean(dim=(2, 3)), dim=1).cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_slice_uint8, classifier_fn, top_labels=1,
                                             hide_color=0, num_samples=LIME_N_SAMPLES)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=True, num_features=10, hide_rest=False)

    lime_fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(mark_boundaries(img_slice_uint8, mask))
    ax[0].set_title(f"LIME - {pid}")
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title("LIME Positive Regions")
    for a in ax: a.axis('off')
    lime_fig.savefig(os.path.join(SAVE_DIR, "lime", f"{pid}_lime.png"), dpi=150, bbox_inches='tight')
    plt.close(lime_fig)

    # ====================== SHAP ======================
    baselines = torch.zeros_like(input_tensor).repeat(SHAP_NSAMPLES, 1, 1, 1, 1)
    shap_attr = grad_shap.attribute(input_tensor, baselines=baselines, target=3,
                                    n_samples=SHAP_NSAMPLES, stdevs=0.01)

    shap_abs_mean = shap_attr.abs().mean(dim=(2, 3, 4)).squeeze(0).cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.bar(['T1n', 'T1c', 'T2w', 'T2f'], shap_abs_mean, color='skyblue')
    plt.title(f"{pid} - Modality Importance (SHAP)")
    plt.ylabel("Average |SHAP Value|")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(SAVE_DIR, "shap", f"{pid}_shap_modality.png"), dpi=150, bbox_inches='tight')
    plt.close()

    shap_results.append({"patient": pid, "T1n": shap_abs_mean[0], "T1c": shap_abs_mean[1],
                         "T2w": shap_abs_mean[2], "T2f": shap_abs_mean[3]})

    # ====================== Faithfulness ======================
    slice_idx = img_np.shape[-1] // 2
    input_slice = input_tensor[..., slice_idx].clone().requires_grad_()
    attr_slice = grad_cam(input_tensor, target_class=3)[..., slice_idx]
    attr_slice = np.expand_dims(attr_slice, axis=0)
    attr_slice = F.interpolate(torch.from_numpy(attr_slice).unsqueeze(0).float(), 
                               size=input_slice.shape[2:], mode='bilinear')[0,0]

    flat_attr = attr_slice.flatten().cpu().numpy()
    sorted_idx = np.argsort(flat_attr)[::-1]

    deletion_scores, insertion_scores = [], []
    baseline = torch.zeros_like(input_slice)
    current = input_slice.clone()
    N_STEPS = 20

    for i in range(N_STEPS):
        frac = i / N_STEPS
        n = int(len(sorted_idx) * frac)

        mask_del = torch.ones_like(current)
        mask_del.view(-1)[sorted_idx[:n]] = 0
        with torch.no_grad():
            score = F.softmax(model((current * mask_del).unsqueeze(0))[..., slice_idx], dim=1)[0,3].item()
        deletion_scores.append(score)

        mask_ins = torch.zeros_like(current)
        mask_ins.view(-1)[sorted_idx[:n]] = 1
        inserted = baseline + (input_slice - baseline) * mask_ins
        with torch.no_grad():
            score = F.softmax(model(inserted.unsqueeze(0))[..., slice_idx], dim=1)[0,3].item()
        insertion_scores.append(score)

    del_auc = auc(np.linspace(0,1,N_STEPS+1), [1.0] + deletion_scores)
    ins_auc = auc(np.linspace(0,1,N_STEPS+1), [0.0] + insertion_scores)

    faith_results.append({"patient": pid, "deletion_auc": del_auc, "insertion_auc": ins_auc})

    print(f"  Avg Dice for {pid}: {np.mean([r['dice'] for r in truth_results[-3:]]):.4f}")

# ========================= SAVE RESULTS =========================
df_truth = pd.DataFrame(truth_results)
df_truth.to_csv(os.path.join(SAVE_DIR, "truthfulness_test_metrics.csv"), index=False)

print("\n" + "="*80)
print("XAI ANALYSIS ON TEST DATASET COMPLETE")
print("="*80)
print("Truthfulness (Dice) Summary on Test Set:")
print(df_truth.groupby("region")["dice"].mean().round(4))
print(f"\nAll results saved in: {SAVE_DIR}")