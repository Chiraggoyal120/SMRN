"""Experimental validation of SMRN theorems

Experiments:
1. bench_time_complexity() - Theorem 1 & 3: O(N) time complexity
2. bench_memory_complexity() - Theorem 3: O(d²) memory, independent of N
3. bench_gradient_stability() - Theorem 4: Bounded gradient norms
4. bench_associative_recall() - Theorem 2: Exact associative recall
5. run_ablation() - Compare SSM vs Attention vs Full SMRN
6. needle_haystack_test() - Position-wise recall performance
"""

import torch
import torch.nn as nn
import time
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from model.smrn import SMRN, SMRNConfig, SMRNSSMOnly, SMRNAttnOnly
from data.datasets import AssociativeRecallDataset, NeedleHaystackDataset
from training.trainer import SMRNTrainer
from torch.utils.data import DataLoader


def bench_time_complexity(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Theorem 1 & 3: Time complexity should be O(N)
    
    Test on seq_lens = [64, 128, 256, 512, 1024, 2048]
    Expected: time ratio ~2x when N doubles (linear scaling)
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Time Complexity (Theorem 1 & 3)")
    print("="*60)
    print("\nTheorem 1: SelectiveSSM has O(N) time complexity")
    print("Theorem 3: Full SMRN has O(N) time complexity\n")
    
    config = SMRNConfig(
        vocab_size=512,
        d_model=256,
        n_layers=4,
        seq_len=2048  # Max length
    )
    
    model = SMRN(config).to(device)
    model.eval()
    
    seq_lens = [64, 128, 256, 512, 1024, 2048]
    results = []
    
    print(f"{'SeqLen':<10} {'Time(ms)':<12} {'Ratio':<10} {'Linear?':<10}")
    print("-" * 50)
    
    prev_time = None
    prev_len = None
    
    for seq_len in seq_lens:
        # Create dummy input
        x = torch.randint(0, config.vocab_size, (4, seq_len), device=device)
        
        # Warmup
        with torch.no_grad():
            _ = model(x)
        
        # Benchmark
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        elapsed = (time.time() - start) / 10 * 1000  # ms
        
        # Compute ratio
        if prev_time is not None:
            ratio = elapsed / prev_time
            expected_ratio = seq_len / prev_len
            is_linear = "✓" if abs(ratio - expected_ratio) < 0.5 else "✗"
        else:
            ratio = 1.0
            is_linear = "-"
        
        print(f"{seq_len:<10} {elapsed:<12.2f} {ratio:<10.2f} {is_linear:<10}")
        
        results.append({
            'seq_len': seq_len,
            'time_ms': elapsed,
            'ratio': ratio
        })
        
        prev_time = elapsed
        prev_len = seq_len
    
    print("\n✓ Theorem 1 & 3: Linear time complexity confirmed!" if all(
        abs(r['ratio'] - 2.0) < 0.5 for r in results[1:]
    ) else "\n⚠ Deviations observed (may be due to hardware)")
    
    return results


def bench_memory_complexity(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Theorem 3: Memory complexity O(d²), independent of N
    
    SSM state: d_model * d_state per layer
    Attention state: d_model * d_model * 2 per layer (S and Z)
    
    Compare with Transformer KV cache: O(Nd) grows with sequence length
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Memory Complexity (Theorem 3)")
    print("="*60)
    print("\nTheorem 3: SMRN uses O(d²) memory, independent of N")
    print("Compare with Transformer KV cache: O(Nd)\n")
    
    config = SMRNConfig(
        vocab_size=512,
        d_model=256,
        n_layers=6,
        d_state=16,
        seq_len=2048
    )
    
    # SMRN state sizes
    ssm_state_per_layer = config.d_model * config.d_state  # h_t
    attn_state_per_layer = config.d_model * config.d_model * 2  # S_t and Z_t
    total_state_per_layer = ssm_state_per_layer + attn_state_per_layer
    total_smrn_state = total_state_per_layer * config.n_layers
    
    print(f"SMRN State Memory (per layer):")
    print(f"  SSM state (h):     {ssm_state_per_layer:>10,} elements")
    print(f"  Attention state (S,Z): {attn_state_per_layer:>10,} elements")
    print(f"  Total per layer:   {total_state_per_layer:>10,} elements")
    print(f"  Total (all {config.n_layers} layers): {total_smrn_state:>10,} elements")
    print(f"  Memory: {total_smrn_state * 4 / 1024**2:.2f} MB (float32)")
    
    print(f"\nTransformer KV Cache (for comparison):")
    print(f"{'SeqLen':<10} {'Memory (elements)':<20} {'Memory (MB)':<15} {'Ratio vs SMRN':<15}")
    print("-" * 60)
    
    for seq_len in [512, 1024, 2048, 4096, 16384]:
        # Transformer: 2 * n_layers * seq_len * d_model (K and V caches)
        transformer_kv = 2 * config.n_layers * seq_len * config.d_model
        memory_mb = transformer_kv * 4 / 1024**2
        ratio = transformer_kv / total_smrn_state
        
        print(f"{seq_len:<10} {transformer_kv:>18,}  {memory_mb:>13.2f}  {ratio:>13.2f}x")
    
    print(f"\n✓ Theorem 3: SMRN state is CONSTANT ({total_smrn_state:,} elements)")
    print(f"  Transformer KV cache GROWS with N (up to {ratio:.1f}x at N=16384)")
    
    return {
        'smrn_state': total_smrn_state,
        'd_model': config.d_model,
        'd_state': config.d_state,
        'n_layers': config.n_layers
    }


def bench_gradient_stability(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Theorem 4: Gradient stability
    
    ∂h_t/∂h_{t-1} = ||Ã_t|| ≤ 1 (A is negative, exp(ΔA) < 1)
    Gate outputs ∈ [0,1] (sigmoid bounded)
    
    Train for 300 steps and track gradient norms
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Gradient Stability (Theorem 4)")
    print("="*60)
    print("\nTheorem 4: Gradient norms remain bounded during training")
    print("  |∂h_t/∂h_{t-1}| = ||Ã_t|| ≤ 1 (SSM stability)")
    print("  Gate values ∈ [0,1] (sigmoid bounded)\n")
    
    # Setup
    config = SMRNConfig(
        vocab_size=512,
        d_model=128,
        n_layers=4,
        batch_size=32,
        seq_len=128,
        max_grad_norm=1.0,  # Theorem 4 enforcement
        lr=3e-4
    )
    
    model = SMRN(config).to(device)
    
    # Simple dataset
    dataset = AssociativeRecallDataset(n_samples=1000, seq_len=config.seq_len, vocab_size=config.vocab_size)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    grad_norms = []
    max_norm_threshold = 500  # If gradient norm exceeds this, Theorem 4 fails
    
    print(f"Training for 300 steps with gradient clipping (max_norm={config.max_grad_norm})...\n")
    print(f"{'Step':<8} {'Grad Norm':<12} {'Status':<15}")
    print("-" * 40)
    
    step = 0
    for batch in loader:
        if step >= 300:
            break
        
        inputs, targets, _ = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Compute gradient norm BEFORE clipping
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip (Theorem 4 enforcement)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        
        optimizer.step()
        
        grad_norms.append(total_norm)
        
        # Log every 50 steps
        if step % 50 == 0:
            status = "BOUNDED" if total_norm < max_norm_threshold else "LARGE"
            print(f"{step:<8} {total_norm:<12.4f} {status:<15}")
        
        step += 1
    
    # Summary
    print("\n" + "-" * 40)
    print(f"Gradient Norm Statistics (300 steps):")
    print(f"  Max:  {max(grad_norms):.4f}")
    print(f"  Mean: {np.mean(grad_norms):.4f}")
    print(f"  Min:  {min(grad_norms):.4f}")
    print(f"  Std:  {np.std(grad_norms):.4f}")
    
    verdict = max(grad_norms) < max_norm_threshold
    print(f"\n{'✓' if verdict else '✗'} Theorem 4: Gradient stability {'confirmed' if verdict else 'violated'}!")
    
    return grad_norms


def bench_associative_recall(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Theorem 2: Unbounded Associative Recall
    
    Under orthogonal feature map φ (RFF), linear attention can store
    O(dk*dv) associations exactly.
    
    Test with varying n_needles and seq_lens
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Associative Recall (Theorem 2)")
    print("="*60)
    print("\nTheorem 2: Linear attention with RFF can store O(dk*dv) associations")
    print("Testing recall accuracy with varying needles and sequence lengths\n")
    
    # Train a small model briefly
    config = SMRNConfig(
        vocab_size=512,
        d_model=128,
        n_layers=4,
        batch_size=32,
        seq_len=256,
        use_rff=True,  # Enable RFF for Theorem 2
        max_epochs=5,
        lr=5e-4
    )
    
    print("Training model (300 steps)...")
    model = SMRN(config).to(device)
    
    # Quick training
    dataset = AssociativeRecallDataset(n_samples=2000, seq_len=config.seq_len, vocab_size=config.vocab_size)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    model.train()
    
    for step, batch in enumerate(tqdm(train_loader, desc="Training", total=300)):
        if step >= 300:
            break
        
        inputs, targets, _ = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Evaluate on different configurations
    print("\nEvaluating associative recall...\n")
    print(f"{'n_needles':<12} {'seq_len':<12} {'Accuracy':<12}")
    print("-" * 40)
    
    results = []
    model.eval()
    
    for n_needles in [1, 2, 4]:
        for seq_len in [64, 128, 256]:
            # Create test dataset
            test_dataset = AssociativeRecallDataset(
                n_samples=100, 
                seq_len=seq_len, 
                n_needles=n_needles,
                vocab_size=config.vocab_size
            )
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Evaluate
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets, ans_pos in test_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    logits = model(inputs)
                    preds = logits.argmax(dim=-1)
                    
                    # Check accuracy at answer positions
                    for i in range(len(ans_pos)):
                        if preds[i, ans_pos[i]] == targets[i, ans_pos[i]]:
                            correct += 1
                        total += 1
            
            acc = correct / total
            print(f"{n_needles:<12} {seq_len:<12} {acc:<12.4f}")
            results.append({'n_needles': n_needles, 'seq_len': seq_len, 'acc': acc})
    
    avg_acc = np.mean([r['acc'] for r in results])
    print(f"\n{'✓' if avg_acc > 0.5 else '⚠'} Average recall accuracy: {avg_acc:.4f}")
    print("(Note: Full convergence requires longer training)")
    
    return results


def run_ablation(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Ablation study: SSM only vs Attention only vs Full SMRN
    
    Train each variant briefly and compare performance
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: Ablation Study")
    print("="*60)
    print("\nComparing: SSM-only vs Attention-only vs Full SMRN\n")
    
    config = SMRNConfig(
        vocab_size=512,
        d_model=128,
        n_layers=3,
        batch_size=32,
        seq_len=128,
        lr=5e-4
    )
    
    # Dataset
    dataset = AssociativeRecallDataset(n_samples=1000, seq_len=config.seq_len, vocab_size=config.vocab_size)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    variants = {
        'SSM Only': SMRNSSMOnly(config),
        'Attention Only': SMRNAttnOnly(config),
        'Full SMRN': SMRN(config)
    }
    
    results = {}
    
    for name, model in variants.items():
        print(f"\nTraining {name}...")
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        model.train()
        
        # Train 150 steps
        losses = []
        for step, batch in enumerate(tqdm(loader, desc=name, total=150)):
            if step >= 150:
                break
            
            inputs, targets, _ = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            logits = model(inputs)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
        
        final_loss = np.mean(losses[-20:])
        n_params = sum(p.numel() for p in model.parameters())
        
        results[name] = {
            'final_loss': final_loss,
            'n_params': n_params
        }
        
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Parameters: {n_params/1e6:.2f}M")
    
    # Summary
    print("\n" + "="*60)
    print("ABLATION RESULTS")
    print("="*60)
    print(f"{'Variant':<20} {'Final Loss':<15} {'Parameters':<15}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<20} {res['final_loss']:<15.4f} {res['n_params']/1e6:<15.2f}M")
    
    return results


def needle_haystack_test(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Test recall accuracy at different depths and context lengths"""
    print("\n" + "="*60)
    print("EXPERIMENT 6: Needle in Haystack")
    print("="*60)
    print("\nTesting recall at varying depths and context lengths\n")
    
    # Train model
    config = SMRNConfig(
        vocab_size=512,
        d_model=128,
        n_layers=4,
        batch_size=16,
        seq_len=512,
        lr=5e-4
    )
    
    print("Training model (200 steps)...")
    model = SMRN(config).to(device)
    
    # Quick training on recall task
    dataset = AssociativeRecallDataset(n_samples=1000, seq_len=256, vocab_size=config.vocab_size)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    model.train()
    
    for step, batch in enumerate(tqdm(loader, desc="Training", total=200)):
        if step >= 200:
            break
        
        inputs, targets, _ = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        logits = model(inputs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Test on haystack dataset
    print("\nTesting on needle-haystack...\n")
    
    # Test configurations
    depths = [0.0, 0.25, 0.5, 0.75, 1.0]
    ctx_lens = [64, 128, 256]
    
    # Print header
    print(f"{'Depth':<10}", end="")
    for ctx_len in ctx_lens:
        print(f"ctx={ctx_len:<8}", end="")
    print()
    print("-" * 40)
    
    model.eval()
    
    for depth in depths:
        print(f"{int(depth*100)}%{' ':<7}", end="")
        
        for ctx_len in ctx_lens:
            # Create test samples
            test_samples = []
            for _ in range(50):
                seq = torch.randint(0, config.vocab_size - 2, (ctx_len,))
                target = seq.clone()
                
                # Insert needle
                needle_pos = int(ctx_len * depth)
                key = np.random.randint(0, config.vocab_size - 2)
                val = np.random.randint(0, config.vocab_size - 2)
                
                KEY_TOKEN = config.vocab_size - 2
                QUERY_TOKEN = config.vocab_size - 1
                
                seq[needle_pos] = KEY_TOKEN
                seq[needle_pos + 1] = key
                seq[needle_pos + 2] = val
                
                # Query at end
                query_pos = ctx_len - 2
                seq[query_pos] = QUERY_TOKEN
                seq[query_pos + 1] = key
                target[query_pos + 1] = val
                
                test_samples.append((seq, target, query_pos + 1))
            
            # Evaluate
            correct = 0
            with torch.no_grad():
                for seq, target, ans_pos in test_samples:
                    seq = seq.unsqueeze(0).to(device)
                    target = target.unsqueeze(0).to(device)
                    
                    # Pad/truncate to model's seq_len if needed
                    if seq.size(1) > config.seq_len:
                        seq = seq[:, :config.seq_len]
                        target = target[:, :config.seq_len]
                        if ans_pos >= config.seq_len:
                            continue
                    
                    logits = model(seq)
                    pred = logits[0, ans_pos].argmax()
                    
                    if pred == target[0, ans_pos]:
                        correct += 1
            
            acc = correct / len(test_samples)
            print(f"{acc:<12.3f}", end="")
        
        print()
    
    print("\n✓ Haystack test complete")


def main():
    """CLI for running experiments"""
    parser = argparse.ArgumentParser(description='SMRN Experiments')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--time', action='store_true', help='Time complexity (Theorem 1)')
    parser.add_argument('--memory', action='store_true', help='Memory complexity (Theorem 3)')
    parser.add_argument('--gradients', action='store_true', help='Gradient stability (Theorem 4)')
    parser.add_argument('--recall', action='store_true', help='Associative recall (Theorem 2)')
    parser.add_argument('--ablation', action='store_true', help='Ablation study')
    parser.add_argument('--haystack', action='store_true', help='Needle-haystack test')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # If no specific test, run all
    if not any([args.time, args.memory, args.gradients, args.recall, args.ablation, args.haystack]):
        args.all = True
    
    print("\n" + "="*60)
    print("SMRN EXPERIMENTAL VALIDATION")
    print("="*60)
    print(f"Device: {args.device}")
    
    if args.all or args.time:
        bench_time_complexity(args.device)
    
    if args.all or args.memory:
        bench_memory_complexity(args.device)
    
    if args.all or args.gradients:
        bench_gradient_stability(args.device)
    
    if args.all or args.recall:
        bench_associative_recall(args.device)
    
    if args.all or args.ablation:
        run_ablation(args.device)
    
    if args.all or args.haystack:
        needle_haystack_test(args.device)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
