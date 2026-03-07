#!/usr/bin/env python3
"""Test numerical stability fixes for SMRN"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.smrn import SMRN, SMRNConfig

def test_no_nan():
    """Test that model doesn't produce NaN"""
    print("\n" + "="*60)
    print("TEST 1: Forward Pass - No NaN/Inf")
    print("="*60)
    
    config = SMRNConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        seq_len=32,
        d_state=8
    )
    
    model = SMRN(config)
    model.eval()
    
    # Test forward pass
    x = torch.randint(0, 100, (4, 16))
    
    with torch.no_grad():
        logits = model(x)
    
    # Check for NaN
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {logits.shape}")
    print(f"✓ Has NaN: {has_nan}")
    print(f"✓ Has Inf: {has_inf}")
    print(f"✓ Output range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
    
    if has_nan:
        print("\n❌ FAILED: Model produces NaN values")
        return False
    elif has_inf:
        print("\n❌ FAILED: Model produces Inf values")
        return False
    else:
        print("\n✅ PASSED: No NaN or Inf values detected")
        return True

def test_training_step():
    """Test training step doesn't produce NaN loss"""
    print("\n" + "="*60)
    print("TEST 2: Training Step - No NaN Loss")
    print("="*60)
    
    config = SMRNConfig(
        vocab_size=50,
        d_model=32,
        n_layers=1,
        seq_len=16
    )
    
    model = SMRN(config)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Test training step
    x = torch.randint(0, 50, (2, 8))
    y = torch.randint(0, 50, (2, 8))
    
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1)
    )
    
    has_nan = torch.isnan(loss).item()
    
    print(f"✓ Loss value: {loss.item():.4f}")
    print(f"✓ Has NaN: {has_nan}")
    
    if not has_nan:
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        grad_has_nan = False
        max_grad = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"❌ NaN gradient in {name}")
                    grad_has_nan = True
                    break
                max_grad = max(max_grad, param.grad.abs().max().item())
        
        if grad_has_nan:
            print("\n❌ FAILED: NaN in gradients")
            return False
        else:
            print(f"✓ Max gradient: {max_grad:.4f}")
            print("\n✅ PASSED: No NaN in loss or gradients")
            return True
    else:
        print("\n❌ FAILED: NaN loss detected")
        return False

def test_multiple_steps():
    """Test multiple training steps for stability"""
    print("\n" + "="*60)
    print("TEST 3: Multiple Steps - Numerical Stability")
    print("="*60)
    
    config = SMRNConfig(
        vocab_size=50,
        d_model=32,
        n_layers=2,
        seq_len=16
    )
    
    model = SMRN(config)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    losses = []
    for step in range(10):
        x = torch.randint(0, 50, (2, 8))
        y = torch.randint(0, 50, (2, 8))
        
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )
        
        if torch.isnan(loss).item() or torch.isinf(loss).item():
            print(f"❌ NaN/Inf at step {step}: {loss.item()}")
            return False
        
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    print(f"✓ Completed 10 training steps")
    print(f"✓ Loss range: [{min(losses):.4f}, {max(losses):.4f}]")
    print(f"✓ Mean loss: {sum(losses)/len(losses):.4f}")
    print("\n✅ PASSED: Stable training for 10 steps")
    return True

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔍 SMRN NUMERICAL STABILITY TEST SUITE")
    print("="*70)
    print("\nTesting fixes:")
    print("  1. SelectiveSSM: delta/dA clamping, NaN checks")
    print("  2. LinearAttention: epsilon=1e-3, state clamping")
    print("  3. EntropyGate: epsilon=1e-4, entropy clamping")
    print("  4. SMRNBlock: LayerNorm before inputs")
    print("  5. SMRN: nan_to_num after layers")
    print("  6. Trainer: NaN detection, loss clamping")
    
    test1 = test_no_nan()
    test2 = test_training_step()
    test3 = test_multiple_steps()
    
    print("\n" + "="*70)
    print("📊 TEST RESULTS")
    print("="*70)
    print(f"  Test 1 (Forward Pass):      {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"  Test 2 (Training Step):      {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"  Test 3 (Multiple Steps):     {'✅ PASS' if test3 else '❌ FAIL'}")
    print("="*70)
    
    if test1 and test2 and test3:
        print("\n🎉 ALL TESTS PASSED - Numerical stability fixes working!")
        print("\n✓ No NaN values in forward pass")
        print("✓ No NaN values in backward pass")
        print("✓ Stable training over multiple steps")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - Review fixes needed")
        sys.exit(1)
