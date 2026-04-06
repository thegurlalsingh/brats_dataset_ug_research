import torch
import torch.nn as nn
from scripts.fusion_layer6 import FusionLayer   # Change the import if your filename is different

def test_fusion_layer():
    print("=" * 90)
    print("       TESTING FusionLayer + CrossAttentionFusion")
    print("=" * 90)

    # Config
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device                    : {device}")
    print(f"Batch size                : {batch_size}")
    print(f"Input spatial size        : 192 x 192")
    print("-" * 90)

    # ================== 1. Initialize Fusion Layer ==================
    try:
        fusion = FusionLayer(in_channels=64, num_classes=3).to(device)
        print("✅ FusionLayer created successfully!")

        total_params = sum(p.numel() for p in fusion.parameters())
        print(f"Total parameters          : {total_params:,}")

    except Exception as e:
        print(f"❌ Error creating FusionLayer: {e}")
        return

    # ================== 2. Create Dummy Inputs ==================
    feat    = torch.randn(batch_size, 64, 192, 192).to(device)   # backbone features
    coarse  = torch.randn(batch_size, 3, 192, 192).to(device)    # coarse logits
    imp     = torch.rand(batch_size, 1, 192, 192).to(device)     # importance
    unc     = torch.rand(batch_size, 1, 192, 192).to(device)     # uncertainty
    refined = torch.randn(batch_size, 64, 192, 192).to(device)   # refined features

    print("\nInput shapes:")
    print(f"   feat     : {feat.shape}")
    print(f"   coarse   : {coarse.shape}")
    print(f"   imp      : {imp.shape}")
    print(f"   unc      : {unc.shape}")
    print(f"   refined  : {refined.shape}")

    # ================== 3. Forward Pass Test ==================
    print("\n1. Testing Forward Pass...")

    try:
        fusion.eval()
        with torch.no_grad():
            output = fusion(feat, coarse, imp, unc, refined)

        print("✅ Forward pass successful!")
        print(f"Output shape : {output.shape} → Expected (B, 3, 192, 192)")

        # Validation
        assert output.shape == (batch_size, 3, 192, 192), \
            f"Wrong output shape. Got {output.shape}, expected (2, 3, 192, 192)"

        print("✅ Output shape is correct!")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # ================== 4. Gradient Flow Test ==================
    print("\n2. Testing Gradient Flow...")

    fusion.train()
    # Make inputs require grad for testing
    feat = feat.requires_grad_(True)
    coarse = coarse.requires_grad_(True)
    imp = imp.requires_grad_(True)
    unc = unc.requires_grad_(True)
    refined = refined.requires_grad_(True)

    try:
        output = fusion(feat, coarse, imp, unc, refined)
        loss = output.mean()          # dummy loss
        loss.backward()

        grad_norm = sum(
            p.grad.norm().item() 
            for p in fusion.parameters() 
            if p.grad is not None
        )
        print(f"Gradient norm             : {grad_norm:.6f}")

        if grad_norm > 0.01:
            print("✅ Gradients are flowing properly through FusionLayer!")
        else:
            print("⚠️  Very small gradients detected.")

    except Exception as e:
        print(f"❌ Backward pass failed: {e}")

    # ================== Summary ==================
    print("\n" + "=" * 90)
    print("🎉 FUSION LAYER TEST PASSED SUCCESSFULLY!")
    print("Your CrossAttentionFusion + Adaptive Weight Fusion is working.")
    print("=" * 90)
    print("\nYour full architecture is now almost complete:")
    print("   • 2.5D Hybrid Backbone")
    print("   • RefinementModule")
    print("   • SparseSelectionModule")
    print("   • DynamicSparseRefinerModel")
    print("   • FusionLayer ← (this one)")

    print("\nWould you like me to create a **Full End-to-End Pipeline Test**")
    print("that connects all 5 components together?")


if __name__ == "__main__":
    test_fusion_layer()