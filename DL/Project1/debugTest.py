import torch
import torch.nn as nn

from src.model.transformer import scaled_dot_product_attention, MultiHeadAttention, EncoderBlock, TransformerEncoder
from src.model.layers import PatchingLayer, PatchEmbedding, InstanceNormalization
from src.model.patch_tst import PatchTST

def test_scaled_dot_product_attention():
    print("--- Testing Scaled Dot-Product Attention ---")
    batch_size, n_heads, seq_len, d_k = 32, 8, 12, 16
    q = k = v = torch.randn(batch_size, n_heads, seq_len, d_k)
    
    output, weights = scaled_dot_product_attention(q, k, v)
    
    assert output.shape == (batch_size, n_heads, seq_len, d_k)
    assert weights.shape == (batch_size, n_heads, seq_len, seq_len)
    print(f"✅ Success: Attention output shape {output.shape} is correct!")

def test_multi_head_attention():
    print("\n--- Testing Multi-Head Attention ---")
    batch_size, seq_len, d_model = 16, 24, 128
    mha = MultiHeadAttention(d_model, n_heads=8)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = mha(x, x, x)
    
    assert output.shape == (batch_size, seq_len, d_model)
    print(f"✅ Success: MHA output shape {output.shape} is correct!")

def test_transformer_stack():
    print("\n--- Testing Transformer Encoder Stack ---")
    batch_size, seq_len, d_model = 8, 24, 128
    # Test single block
    block = EncoderBlock(d_model, n_heads=8, d_ff=512)
    x = torch.randn(batch_size, seq_len, d_model)
    assert block(x).shape == (batch_size, seq_len, d_model)
    
    # Test full stack (3 layers)
    encoder = TransformerEncoder(d_model, n_heads=8, d_ff=512, n_layers=3)
    output = encoder(x)
    assert output.shape == (batch_size, seq_len, d_model)
    print(f"✅ Success: Transformer Stack (3 layers) processed correctly!")

def test_patching_and_normalization():
    print("\n--- Testing Patching and Instance Normalization ---")
    batch_size, n_vars, lookback = 8, 1, 96
    patch_len, stride = 16, 8
    
    x = torch.randn(batch_size, n_vars, lookback)
    
    # Test Norm
    revin = InstanceNormalization()
    x_norm, stats = revin(x, mode='norm')
    assert x_norm.shape == x.shape
    
    # Test Patching
    patcher = PatchingLayer(patch_len, stride)
    patches = patcher(x_norm)
    
    num_patches = (lookback - patch_len) // stride + 1
    assert patches.shape == (batch_size, num_patches, patch_len)
    print(f"✅ Success: Generated {num_patches} patches correctly after normalization!")

def test_full_patch_tst_integration():
    print("\n--- Testing Full PatchTST Integration (The Big One) ---")
    # Configuration typical for Traffic dataset
    config = {
        "n_channels": 7,
        "lookback_len": 96,
        "patch_len": 16,
        "stride": 8,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 256,
        "n_layers": 3,
        "pred_len": 24
    }
    
    model = PatchTST(**config)
    
    # Input: [Batch, Channels, Lookback]
    x = torch.randn(16, config["n_channels"], config["lookback_len"])
    
    try:
        output = model(x)
        # Expected output: [Batch, Channels, Pred_Len]
        assert output.shape == (16, config["n_channels"], config["pred_len"])
        print(f"✅ Success: Full PatchTST integration works!")
        print(f"   Input shape: {x.shape} -> Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ Failed: Error in PatchTST forward pass: {e}")
        raise e

if __name__ == "__main__":
    print("🚀 STARTING ALL MODEL TESTS\n" + "="*30)
    test_scaled_dot_product_attention()
    test_multi_head_attention()
    test_transformer_stack()
    test_patching_and_normalization()
    test_full_patch_tst_integration()
    print("\n" + "="*30 + "\n🏆 ALL TESTS PASSED! Architecture is solid.")