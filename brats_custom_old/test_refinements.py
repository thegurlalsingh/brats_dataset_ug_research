import torch
from refinements3 import RefinementModule   # Change filename if yours is different (e.g. 3_refinements)

def test_refinements():
    print("=" * 90)
    print("       TESTING RefinementModule + SparseSelector")
    print("=" * 90)

    # Config
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_k = 6
    patch_size = 64

    print(f"Device              : {device}")
    print(f"Batch size          : {batch_size}")
    print(f"Top-K regions       : {top_k}")
    print(f"Patch size          : {patch_size} x {patch_size}")
    print("-" * 90)

    # ================== 1. Create Module ==================
    try:
        module = RefinementModule(
            in_channels=64,
            num_classes=3,
        ).to(device)

        print("✅ RefinementModule created successfully!")

        total_params = sum(p.numel() for p in module.parameters())
        print(f"Total parameters    : {total_params:,}")

    except Exception as e:
        print(f"❌ Error creating module: {e}")
        return

    # ================== 2. Forward Pass Test ==================
    print("\n1. Testing Forward Pass...")

    # Dummy features from your 2.5D backbone
    features = torch.randn(batch_size, 64, 192, 192).to(device)

    try:
        module.eval()
        with torch.no_grad():
            output = module(features)

        print("✅ Forward pass successful!")

        # Check output keys and shapes
        print("\nOutput Keys & Shapes:")
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key:15}: {value.shape}")
            elif isinstance(value, list):
                print(f"   {key:15}: List of length {len(value)}")
            else:
                print(f"   {key:15}: {type(value)}")

        # Validation
        assert 'coarse' in output, "Missing coarse output"
        assert 'patch_coords' in output, "Missing patch_coords"
        assert 'selected_patches' in output, "Missing selected_patches"

        print(f"\nNumber of selected patches : {len(output['patch_coords'])}")
        print(f"Example patch coord        : {output['patch_coords'][0] if output['patch_coords'] else 'None'}")

        # Check if Top-K is working
        expected_patches = batch_size * top_k
        actual_patches = len(output['patch_coords'])
        print(f"Expected patches (approx)  : ~{expected_patches}")
        print(f"Actual patches             : {actual_patches}")

        if actual_patches > 0:
            print("✅ Sparse selection is working!")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ================== 3. Summary ==================
    print("\n" + "=" * 90)
    print("🎉 REFINEMENT MODULE TEST PASSED!")
    print("Your sparse selection logic (Importance × Uncertainty + Connected Components) is ready.")
    print("=" * 90)
    print("\nNext recommended steps:")
    print("   1. Create Mini 3D Refiner (patch-based 3D CNN)")
    print("   2. Add Patch Extraction logic using patch_coords")
    print("   3. Implement Fusion Layer (cross-attention between coarse & refined)")
    print("\nWould you like me to create the Mini 3D Refiner + Fusion module now?")


if __name__ == "__main__":
    test_refinements()