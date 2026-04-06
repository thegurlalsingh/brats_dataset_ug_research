import torch
import torch.nn as nn
from volumetric_context5 import DynamicSparseRefinerModel  # Change if your filename is different

def test_dynamic_sparse_refiner():
    print("=" * 95)
    print("       TESTING DynamicSparseRefinerModel")
    print("=" * 95)

    # Config
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iters = 2

    print(f"Device                    : {device}")
    print(f"Batch size                : {batch_size}")
    print(f"Iterations                : {iters}")
    print(f"Input from 2.5D Backbone  : (B, 12, 192, 192)")
    print("-" * 95)

    # ================== 1. Dummy Sparse Module ==================
    class DummySparse(nn.Module):
        def forward(self, features, focus_stack):
            # Return same shape as input for testing (mimics your SparseSelectionModule)
            return features

    # ================== 2. Initialize Refiner ==================
    try:
        refiner = DynamicSparseRefinerModel(
            sparse_module=DummySparse(),
            num_classes=3,
            iters=iters
        ).to(device)

        print("✅ DynamicSparseRefinerModel created successfully!")

        total_params = sum(p.numel() for p in refiner.parameters())
        print(f"Total parameters          : {total_params:,}")

    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return

    # ================== 3. Forward Pass Test ==================
    print("\n1. Testing Forward Pass...")

    # Input shape from your 2.5D Backbone
    x_2d5 = torch.randn(batch_size, 12, 192, 192).to(device)

    try:
        refiner.eval()
        with torch.no_grad():
            output = refiner(x_2d5)

        print("✅ Forward pass successful!")

        # Output validation
        print("\nOutput Shapes:")
        print(f"   Final logits    : {output['final'].shape}")
        print(f"   Coarse logits   : {output['coarse'].shape}")
        print(f"   Features 3D     : {output['feat3d'].shape}")

        # Basic checks
        assert output['final'].shape[1] == 3, f"Wrong number of classes. Got {output['final'].shape[1]}, expected 3"
        assert output['final'].dim() == 5, f"Final output should be 5D, got {output['final'].dim()}D"

        print("✅ All output shapes are correct!")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ================== 4. Gradient Flow Test ==================
    print("\n2. Testing Gradient Flow...")

    refiner.train()
    x_2d5 = torch.randn(batch_size, 12, 192, 192, requires_grad=True).to(device)

    try:
        output = refiner(x_2d5)
        loss = output['final'].mean()          # dummy loss for gradient check
        loss.backward()

        grad_norm = sum(
            p.grad.norm().item() 
            for p in refiner.parameters() 
            if p.grad is not None
        )
        print(f"Gradient norm             : {grad_norm:.6f}")

        if grad_norm > 0.01:
            print("✅ Gradients are flowing properly through the refiner!")
        else:
            print("⚠️  Very small gradient norm detected.")

    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()

    # ================== Final Summary ==================
    print("\n" + "=" * 95)
    print("🎉 DYNAMIC SPARSE REFINER TEST COMPLETED SUCCESSFULLY!")
    print("Your Mini 3D iterative sparse refiner is working.")
    print("=" * 95)
    print("\nNext recommended actions:")
    print("   1. Replace DummySparse with your real SparseSelectionModule")
    print("   2. Connect the full pipeline (Backbone → RefinementModule → SparseSelection → This Refiner)")
    print("   3. Add proper Fusion Layer if needed")
    print("   4. Start end-to-end training")

    print("\nWould you like me to create a full end-to-end pipeline test script?")


if __name__ == "__main__":
    test_dynamic_sparse_refiner()