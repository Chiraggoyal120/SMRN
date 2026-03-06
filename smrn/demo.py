#!/usr/bin/env python
"""Quick demo of SMRN capabilities"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.smrn import SMRN, SMRNConfig
from data.datasets import AssociativeRecallDataset
from torch.utils.data import DataLoader

def demo_architecture():
    """Demonstrate SMRN architecture"""
    print("\n" + "="*70)
    print("SMRN ARCHITECTURE DEMO")
    print("="*70)
    
    config = SMRNConfig(
        vocab_size=512,
        d_model=256,
        n_layers=4,
        d_state=16,
        seq_len=128,
        use_rff=True
    )
    
    model = SMRN(config)
    
    print("\n📐 Model Configuration:")
    print(f"   Vocabulary size: {config.vocab_size}")
    print(f"   Model dimension (d_model): {config.d_model}")
    print(f"   Number of layers: {config.n_layers}")
    print(f"   SSM state dimension: {config.d_state}")
    print(f"   Window size (entropy): {config.window_size}")
    print(f"   Random Fourier Features: {config.use_rff}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Breakdown by component
    print("\n🔧 Component Breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"   {name:15s}: {params:>10,} params")
    
    print("\n⚡ Complexity Analysis:")
    print(f"   Time complexity: O(N) per sequence")
    print(f"   Memory complexity: O(d²) = O({config.d_model}²) independent of N")
    
    # SSM state size
    ssm_state_per_layer = config.d_model * config.d_state
    attn_state_per_layer = config.d_model * config.d_model * 2
    total_state = (ssm_state_per_layer + attn_state_per_layer) * config.n_layers
    
    print(f"\n💾 Memory Footprint:")
    print(f"   SSM state per layer: {ssm_state_per_layer:,} elements")
    print(f"   Attention state per layer: {attn_state_per_layer:,} elements")
    print(f"   Total state (all layers): {total_state:,} elements")
    print(f"   State memory: {total_state * 4 / 1024**2:.2f} MB (float32)")

def demo_forward_pass():
    """Demonstrate forward pass with gate visualization"""
    print("\n" + "="*70)
    print("FORWARD PASS WITH GATE VISUALIZATION")
    print("="*70)
    
    config = SMRNConfig(vocab_size=256, d_model=128, n_layers=4, seq_len=64)
    model = SMRN(config)
    model.eval()
    
    # Create input
    batch_size = 2
    seq_len = 32
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\n📥 Input: {x.shape} (batch={batch_size}, seq_len={seq_len})")
    
    # Forward pass with gate values
    with torch.no_grad():
        logits, gate_values = model(x, return_gate_values=True)
    
    print(f"📤 Output: {logits.shape} (batch, seq_len, vocab_size)")
    
    print(f"\n🚪 Gate Values (per layer):")
    print("   (High g → SSM dominant, Low g → Attention dominant)")
    print()
    
    for i, gates in enumerate(gate_values):
        avg_gate = gates.mean().item()
        min_gate = gates.min().item()
        max_gate = gates.max().item()
        
        # Visual bar
        bar_len = int(avg_gate * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        
        if avg_gate > 0.6:
            pathway = "SSM-dominant"
        elif avg_gate > 0.4:
            pathway = "Balanced"
        else:
            pathway = "Attention-dominant"
        
        print(f"   Layer {i+1}: [{bar}] {avg_gate:.3f} ({pathway})")
        print(f"           Range: [{min_gate:.3f}, {max_gate:.3f}]")

def demo_associative_recall():
    """Demonstrate associative recall capability"""
    print("\n" + "="*70)
    print("ASSOCIATIVE RECALL DEMONSTRATION")
    print("="*70)
    print("\nTheorem 2: Linear attention can store O(dk·dv) associations exactly")
    print("Testing with key-value pairs...\n")
    
    config = SMRNConfig(
        vocab_size=256,
        d_model=128,
        n_layers=4,
        seq_len=64,
        use_rff=True
    )
    
    model = SMRN(config)
    
    # Quick training (10 steps)
    print("🔄 Quick training (10 steps)...")
    dataset = AssociativeRecallDataset(n_samples=50, seq_len=config.seq_len, vocab_size=config.vocab_size)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    model.train()
    for step, (inputs, targets, _) in enumerate(loader):
        if step >= 10:
            break
        
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Test recall
    print("🧪 Testing recall on 5 examples...\n")
    model.eval()
    
    KEY_TOKEN = config.vocab_size - 2
    QUERY_TOKEN = config.vocab_size - 1
    
    correct = 0
    for trial in range(5):
        # Create sequence with key-value pairs
        seq = torch.randint(0, config.vocab_size - 2, (1, config.seq_len))
        
        # Insert 2 pairs
        pairs = [(10, 20), (30, 40)]
        positions = [10, 30]
        
        for (key, val), pos in zip(pairs, positions):
            seq[0, pos] = KEY_TOKEN
            seq[0, pos + 1] = key
            seq[0, pos + 2] = val
        
        # Query
        query_key, query_val = pairs[0]
        query_pos = config.seq_len - 2
        seq[0, query_pos] = QUERY_TOKEN
        seq[0, query_pos + 1] = query_key
        
        # Predict
        with torch.no_grad():
            logits = model(seq)
            pred = logits[0, query_pos + 1].argmax().item()
            probs = torch.softmax(logits[0, query_pos + 1], dim=0)
            confidence = probs[pred].item()
        
        is_correct = (pred == query_val)
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"   Trial {trial+1}: key={query_key:2d} → expected={query_val:2d}, "
              f"predicted={pred:2d} (conf={confidence:.2f}) [{status}]")
    
    print(f"\n📊 Accuracy: {correct}/5 = {correct/5*100:.0f}%")
    print("   (Note: Full performance requires more training)")

def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("🚀 SMRN: Selective Memory and Recall Network")
    print("="*70)
    print("\nPaper: 'Selective Memory and Recall Network: A Theoretical Analysis")
    print("       for Linear-Time Agents'")
    print("Authors: Chirag Goyal, Manoj Kumar — VIT Bhopal University\n")
    
    demo_architecture()
    demo_forward_pass()
    demo_associative_recall()
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)
    print("\n📚 Full capabilities:")
    print("   • O(N) time complexity (Theorem 1 & 3)")
    print("   • O(d²) memory, independent of N (Theorem 3)")
    print("   • Unbounded associative recall with RFF (Theorem 2)")
    print("   • Gradient stability via log-parameterized A (Theorem 4)")
    print("   • Dynamic pathway selection via entropy gating")
    print("\n📖 For more details, see: /app/smrn/README.md")
    print("🧪 Run experiments: python experiments/run_experiments.py --all")

if __name__ == '__main__':
    main()
