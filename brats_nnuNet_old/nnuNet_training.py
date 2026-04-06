import os
from pathlib import Path
import shutil
import json
from tqdm import tqdm

# ================== CONFIG ==================
brats_folder = Path(r"C:\Users\Gurlal-Stu\Downloads\brats\data")          # ← Change if needed
dataset_name = "Dataset001_BraTS2023"

# nnU-Net environment paths
os.environ['nnUNet_raw']        = r"C:\Users\Gurlal-Stu\Downloads\brats\output\nnUNet_raw"
os.environ['nnUNet_preprocessed'] = r"C:\Users\Gurlal-Stu\Downloads\brats\output\nnUNet_preprocessed"
os.environ['nnUNet_results']    = r"C:\Users\Gurlal-Stu\Downloads\brats\output\nnUNet_results"

raw_folder = Path(os.environ['nnUNet_raw']) / dataset_name

# Create folders
(raw_folder / "imagesTr").mkdir(parents=True, exist_ok=True)
(raw_folder / "labelsTr").mkdir(parents=True, exist_ok=True)

print(f"✅ nnU-Net folders ready at: {raw_folder}\n")

# ================== MODALITY MAP ==================
modality_map = {
    "t1n": "0000",
    "t1c": "0001",
    "t2w": "0002",
    "t2f": "0003"
}

# ================== COPY DATA ==================
patient_dirs = sorted([d for d in brats_folder.iterdir() if d.is_dir()])

processed_count = 0
print(f"Found {len(patient_dirs)} patient folders. Starting copy...")

for patient in tqdm(patient_dirs, desc="Processing patients"):
    patient_name = patient.name
    seg_file = patient / f"{patient_name}-seg.nii.gz"
    
    if not seg_file.exists():
        print(f"⚠ Skipping {patient_name} (no segmentation file)")
        continue
    
    case_id = patient_name.replace("BraTS-GLI-", "").replace("-000", "")
    
    # Copy 4 modalities
    for suffix, channel in modality_map.items():
        src = patient / f"{patient_name}-{suffix}.nii.gz"
        if src.exists():
            dst = raw_folder / "imagesTr" / f"{case_id}_{channel}.nii.gz"
            shutil.copy2(src, dst)
        else:
            print(f"⚠ Missing modality: {suffix} for {patient_name}")
    
    # Copy segmentation
    shutil.copy2(seg_file, raw_folder / "labelsTr" / f"{case_id}.nii.gz")
    processed_count += 1

print(f"\n✅ {processed_count} patients copied successfully!\n")

# ================== CREATE dataset.json ==================
dataset_json = {
    "channel_names": {str(i): m for i, m in enumerate(["T1", "T1c", "T2", "FLAIR"])},
    "labels": {
        "background": 0,
        "NCR_NET": 1,
        "ED": 2,
        "ET": 3
    },
    "numTraining": processed_count,
    "file_ending": ".nii.gz"
}

with open(raw_folder / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

print("✅ dataset.json created!\n")

# ================== RUN nnU-Net COMMANDS ==================
print("🚀 Installing/Updating nnU-Net...")
os.system("pip install nnunetv2")

print("\n🚀 Running planning and preprocessing...")
os.system("nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity")

print("\n🚀 Starting training (fold 0)...")
os.system("nnUNetv2_train 001 3d_fullres 0 --c")

print("\n🎉 Script finished!")