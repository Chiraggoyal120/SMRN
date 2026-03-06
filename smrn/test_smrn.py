#!/usr/bin/env python
"""Quick test script to verify SMRN implementation"""

import torch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from model.smrn import SMRN, SMRNConfig, SMRNSSMOnly, SMRNAttnOnly
from data.datasets import AssociativeRecallDataset, CharLMDataset
from torch.utils.data import DataLoader

def test_model():
    """Test model forward pass"""
    print("\n" + "="*60)
    print("TEST 1: Model Forward Pass")
    print("="*60)
    
    config = SMRNConfig(
        vocab_size=512,
        d_model=128,
        n_layers=3,
        seq_len=64
    )
    
    model = SMRN(config)
    print(f"✓ Created SMRN with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Test forward
    x = torch.randint(0, config.vocab_size, (4, 32))
    logits = model(x)
    assert logits.shape == (4, 32, config.vocab_size)
    print(f"✓ Forward pass: {x.shape} -> {logits.shape}")
    
    # Test with gate values
    logits, gates = model(x, return_gate_values=True)
    assert len(gates) == config.n_layers
    print(f"✓ Gate values: {len(gates)} layers")
    
    # Test generation
    generated = model.generate(x[:1], max_new_tokens=10, temperature=1.0)
    assert generated.shape[1] == x.shape[1] + 10
    print(f"✓ Generation: {x[:1].shape} -> {generated.shape}")
    
    print("✓ Model test PASSED\n")

def test_ablations():
    """Test ablation models"""
    print("="*60)
    print("TEST 2: Ablation Models")
    print("="*60)
    
    config = SMRNConfig(vocab_size=256, d_model=64, n_layers=2, seq_len=32)
    
    # SSM Only
    model_ssm = SMRNSSMOnly(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    logits = model_ssm(x)
    assert logits.shape == (2, 16, config.vocab_size)
    print(f"✓ SSM Only: {x.shape} -> {logits.shape}")
    
    # Attention Only
    model_attn = SMRNAttnOnly(config)
    logits = model_attn(x)
    assert logits.shape == (2, 16, config.vocab_size)
    print(f"✓ Attention Only: {x.shape} -> {logits.shape}")
    
    print("✓ Ablation test PASSED\n")

def test_datasets():
    """Test datasets"""
    print("="*60)
    print("TEST 3: Datasets")
    print("="*60)
    
    # Associative Recall
    ds = AssociativeRecallDataset(n_samples=10, seq_len=32, vocab_size=128)
    loader = DataLoader(ds, batch_size=4)
    inputs, targets, ans_pos = next(iter(loader))
    assert inputs.shape == (4, 32)
    print(f"✓ AssociativeRecall: batch shape {inputs.shape}")
    
    # Char LM
    text = "Hello world! This is a test." * 10
    ds = CharLMDataset(text, seq_len=32)
    loader = DataLoader(ds, batch_size=2)
    inputs, targets = next(iter(loader))
    assert inputs.shape == targets.shape
    print(f"✓ CharLM: vocab_size={ds.vocab_size}, batch shape {inputs.shape}")
    
    print("✓ Dataset test PASSED\n")

def test_training_step():
    """Test one training step"""
    print("="*60)
    print("TEST 4: Training Step")
    print("="*60)
    
    config = SMRNConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        seq_len=32,
        batch_size=4
    )
    
    model = SMRN(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create data
    ds = AssociativeRecallDataset(n_samples=20, seq_len=config.seq_len, vocab_size=config.vocab_size)
    loader = DataLoader(ds, batch_size=config.batch_size)
    
    # Training step
    model.train()
    inputs, targets, _ = next(iter(loader))
    
    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )
    
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients exist
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads
    
    # Gradient clipping (Theorem 4)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Grad norm: {grad_norm.item():.4f}")
    print(f"✓ Gradients computed and clipped")
    print("✓ Training step PASSED\n")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("SMRN IMPLEMENTATION TEST SUITE")
    print("="*60)
    
    try:
        test_model()
        test_ablations()
        test_datasets()
        test_training_step()
        
        print("="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nImplementation is ready!")
        print("\nNext steps:")
        print("  1. Train: python training/trainer.py --task recall")
        print("  2. Validate: python experiments/run_experiments.py --all")
        print("  3. Visualize: python utils/visualize.py")
        print("  4. Generate: python inference/generate.py --ckpt <path>")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
