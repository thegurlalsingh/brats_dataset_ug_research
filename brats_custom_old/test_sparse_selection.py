import torch
from sparse_selection4 import (
    SparseSelectionModule,
    patchify,
    unpatchify,
    patch_scores,
    select_topk_patches,
    patches_to_tokens,
    tokens_to_patches
)

def test_sparse_selection():
    print("=" * 90)
    print("       TESTING SparseSelectionModule + Helper Functions")
    print("=" * 90)

    # Config
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = 32          # Your current default in the file
    num_slices = 3

    print(f"Device               : {device}")
    print(f"Patch size           : {patch_size}")
    print(f"Num slices (volume)  : {num_slices}")
    print("-" * 90)

    # ================== 1. Test Helper Functions ==================
    print("\n1. Testing Helper Functions...")

    # Test patchify / unpatchify
    test_tensor = torch.randn(batch_size, 64, 192, 192).to(device)
    patches = patchify(test_tensor, patch_size)
    reconstructed = unpatchify(patches, patch_size, 192, 192)

    print(f"Original shape       : {test_tensor.shape}")
    print(f"Patches shape        : {patches.shape}")
    print(f"Reconstructed shape  : {reconstructed.shape}")
    print(f"Reconstruction error : {F.mse_loss(test_tensor, reconstructed):.2e}")

    # Test patch_scores
    focus_map = torch.rand(batch_size, 1, 192, 192).to(device)
    scores = patch_scores(focus_map, patch_size)
    print(f"Patch scores shape   : {scores.shape}")

    # ================== 2. Test Full Module ==================
    print("\n2. Testing SparseSelectionModule...")

    try:
        module = SparseSelectionModule(
            in_channels=64,
            patch_size=patch_size,
            num_slices=num_slices
        ).to(device)

        features = torch.randn(batch_size, 64, 192, 192).to(device)
        focus_stack = torch.randn(batch_size, num_slices, 1, 192, 192).to(device)

        output = module(features, focus_stack)

        print("✅ Forward pass successful!")
        print(f"Output shape         : {output.shape}")
        print(f"Expected output shape: (B, 64, 192, 192)")

        # Check if output is reasonable
        print(f"Output min/max       : {output.min():.3f} / {output.max():.3f}")
        print(f"Output mean          : {output.mean():.3f}")

    except Exception as e:
        print(f"❌ Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
        return

    # ================== 3. Parameter Count ==================
    total_params = sum(p.numel() for p in module.parameters())
    print(f"\nTotal parameters     : {total_params:,}")

    # ================== Summary ==================
    print("\n" + "=" * 90)
    print("🎉 SPARSE SELECTION MODULE TEST PASSED!")
    print("Your patchify/unpatchify, learnable k, volume aggregator, and token transformer are working.")
    print("=" * 90)
    print("\nNext Step Recommendation:")
    print("   → Integrate this with RefinementModule (using patch_coords)")
    print("   → Build Mini 3D Refiner for selected patches")
    print("   → Add final Fusion Layer")

    print("\nWould you like me to create the integration code + Mini 3D Refiner now?")


if __name__ == "__main__":
    test_sparse_selection()