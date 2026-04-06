import torch
import numpy as np
import matplotlib.pyplot as plt

# Import your dataset
from scripts.dataloader1 import BraTSDataset   # Change this if your filename is different


def test_dataloader():
    # ================== CONFIG ==================
    root_dir = r"C:\Users\Gurlal-Stu\Downloads\custom\data"   # ←←← CHANGE THIS TO YOUR ACTUAL PATH
    k = 3
    
    print(f"Testing BraTSDataset with k={k}, crop=True")
    print("=" * 70)

    # Create dataset
    dataset = BraTSDataset(
        root_dir=root_dir,
        k=k,
        transform=None,
                         # Safe mild cropping
    )

    print(f"✅ Dataset created successfully!")
    print(f"Total samples : {len(dataset)}")
    print(f"Patients found : {len(dataset.patients) if hasattr(dataset, 'patients') else 'N/A'}")

    if len(dataset) == 0:
        print("❌ ERROR: No data found! Please check your root_dir and folder structure.")
        return

    # ================== SINGLE SAMPLE TEST ==================
    print("\n1. Testing single sample...")

    try:
        x, y = dataset[0]
        
        print(f"Input shape  (x) : {x.shape}")      # Expected: (12, H, W) for k=3
        print(f"Target shape (y) : {y.shape}")      # Expected: (3, H, W)
        print(f"Input  → min: {x.min():.3f}, max: {x.max():.3f}, mean: {x.mean():.3f}")
        print(f"Target → unique values: {torch.unique(y)}")

        expected_channels = 4 * k
        if x.shape[0] != expected_channels:
            print(f"⚠️  Warning: Expected {expected_channels} input channels, got {x.shape[0]}")

    except Exception as e:
        print(f"❌ Error while loading sample: {e}")
        return

    # ================== BATCH TEST ==================
    print("\n2. Testing DataLoader batching...")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,          # Set 0 for debugging on Windows
        pin_memory=False,
        drop_last=True
    )

    try:
        x_batch, y_batch = next(iter(dataloader))
        print(f"Batch input shape : {x_batch.shape}")   # (B, 12, H, W)
        print(f"Batch target shape: {y_batch.shape}")   # (B, 3, H, W)
    except Exception as e:
        print(f"❌ Error during batching: {e}")

    # ================== VISUALIZATION ==================
    print("\n3. Visualizing samples...")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"BraTS 2.5D Sample Visualization (k={k})", fontsize=14)

    for i in range(min(4, len(dataset))):
        x, y = dataset[i]
        
        # Take T1c from central slice (index = 4*1 + 1 = 5 for k=3)
        central_mod_idx = 4 * (k // 2) + 1          # T1c of central slice
        img = x[central_mod_idx].numpy()

        # Input image
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title(f"Input (T1c)\nSlice {i}")
        axes[0, i].axis('off')

        # Ground truth overlay
        wt = y[0].numpy()
        et = y[2].numpy()
        
        overlay = np.zeros((wt.shape[0], wt.shape[1], 3))
        overlay[..., 0] = wt      # Red = Whole Tumor
        overlay[..., 1] = et      # Green = Enhancing Tumor

        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].imshow(overlay, alpha=0.55)
        axes[1, i].set_title("GT: WT (red) + ET (green)")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    print("\n✅ Dataloader test completed!")
    print("If cropping looks safe and shapes are correct, we can proceed to model.")


if __name__ == "__main__":
    test_dataloader()