"""Inference and text generation for SMRN

Functions:
- load_model() - Load trained model from checkpoint
- generate_text() - Autoregressive text generation with sampling
- demo_recall() - Demonstrate associative recall
- visualize_gate_behavior() - ASCII visualization of gate values
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from model.smrn import SMRN, SMRNConfig


def load_model(ckpt_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Load SMRN model from checkpoint
    
    Args:
        ckpt_path: Path to checkpoint file
        device: Device to load model on
    Returns:
        model: Loaded SMRN model
        config: Model configuration
    """
    print(f"Loading model from {ckpt_path}...")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']
    
    model = SMRN(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    return model, config


def generate_text(model, prompt: str, char2idx: dict, idx2char: dict,
                  max_tokens: int = 200, temperature: float = 0.8,
                  top_k: int = 40, top_p: float = 0.9,
                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Generate text autoregressively
    
    Args:
        model: SMRN model
        prompt: Initial text prompt
        char2idx: Character to index mapping
        idx2char: Index to character mapping
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k filtering (0 = disabled)
        top_p: Nucleus (top-p) filtering (1.0 = disabled)
        device: Device for generation
    Returns:
        generated_text: Complete generated text
    """
    model.eval()
    
    # Encode prompt
    context = [char2idx.get(ch, 0) for ch in prompt]
    context = torch.tensor(context, dtype=torch.long).unsqueeze(0).to(device)
    
    print(f"Prompt: '{prompt}'")
    print(f"Generating {max_tokens} tokens...\n")
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # Decode
    generated_ids = generated[0].cpu().tolist()
    generated_text = ''.join([idx2char.get(i, '?') for i in generated_ids])
    
    return generated_text


def demo_recall(model, vocab_size: int, n_trials: int = 10, 
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Demonstrate associative recall capability (Theorem 2)
    
    Create key-value pairs, query model, and check if it recalls correctly
    
    Args:
        model: SMRN model
        vocab_size: Vocabulary size
        n_trials: Number of trials
        device: Device for computation
    """
    print("\n" + "="*60)
    print("ASSOCIATIVE RECALL DEMO (Theorem 2)")
    print("="*60)
    print("\nTesting model's ability to recall key-value associations\n")
    
    model.eval()
    KEY_TOKEN = vocab_size - 2
    QUERY_TOKEN = vocab_size - 1
    
    correct = 0
    
    for trial in range(n_trials):
        # Create sequence with key-value pairs
        seq_len = 128
        seq = torch.randint(0, vocab_size - 2, (1, seq_len), device=device)
        
        # Insert 3 key-value pairs
        pairs = []
        positions = [20, 50, 80]
        
        for pos in positions:
            key = np.random.randint(0, vocab_size - 2)
            val = np.random.randint(0, vocab_size - 2)
            pairs.append((key, val))
            
            seq[0, pos] = KEY_TOKEN
            seq[0, pos + 1] = key
            seq[0, pos + 2] = val
        
        # Query one of the pairs
        query_key, query_val = pairs[np.random.randint(0, len(pairs))]
        query_pos = seq_len - 2
        
        seq[0, query_pos] = QUERY_TOKEN
        seq[0, query_pos + 1] = query_key
        
        # Predict
        with torch.no_grad():
            logits = model(seq)
            pred = logits[0, query_pos + 1].argmax().item()
        
        # Check
        is_correct = (pred == query_val)
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        print(f"Trial {trial+1:2d}: Query key={query_key:3d} → Expected val={query_val:3d}, "
              f"Predicted val={pred:3d} [{status}]")
    
    accuracy = correct / n_trials
    print(f"\nAccuracy: {accuracy*100:.1f}% ({correct}/{n_trials})")
    
    if accuracy > 0.7:
        print("✓ Strong associative recall capability!")
    elif accuracy > 0.5:
        print("⚠ Moderate recall (may need more training)")
    else:
        print("✗ Weak recall (model needs more training)")


def visualize_gate_behavior(model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Visualize gate values per layer using ASCII bar chart
    
    Gate value interpretation:
    - g > 0.7: SSM dominant (compression) → '█' (full block)
    - g > 0.4: Balanced → '▓' (medium shade)
    - g ≤ 0.4: Attention dominant (recall) → '░' (light shade)
    
    Args:
        model: SMRN model
        device: Device for computation
    """
    print("\n" + "="*60)
    print("GATE BEHAVIOR VISUALIZATION")
    print("="*60)
    print("\nGate values per layer (averaged over sequence)")
    print("█ = SSM dominant (g > 0.7) | ▓ = Balanced | ░ = Attention dominant (g < 0.4)\n")
    
    model.eval()
    
    # Generate dummy input
    seq_len = 128
    vocab_size = model.config.vocab_size
    x = torch.randint(0, vocab_size, (1, seq_len), device=device)
    
    # Forward with gate values
    with torch.no_grad():
        _, gate_values = model(x, return_gate_values=True)
    
    # Visualize each layer
    for layer_idx, gates in enumerate(gate_values):
        # Average gate value across sequence and dimensions
        avg_gate = gates.mean().item()
        
        # Create ASCII bar
        bar_length = int(avg_gate * 50)
        
        if avg_gate > 0.7:
            bar = '█' * bar_length
            label = "SSM"
        elif avg_gate > 0.4:
            bar = '▓' * bar_length
            label = "BAL"
        else:
            bar = '░' * bar_length
            label = "ATN"
        
        print(f"Layer {layer_idx+1:2d} [{label}] {bar} {avg_gate:.3f}")
    
    print("\nInterpretation:")
    print("  SSM (high g): Model relies on compression pathway (predictable context)")
    print("  ATN (low g):  Model relies on recall pathway (surprising/relevant context)")
    print("  BAL (mid g):  Balanced mixture of both pathways")


def main():
    """CLI for SMRN inference"""
    parser = argparse.ArgumentParser(description='SMRN Inference')
    
    # Model loading
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    
    # Generation
    parser.add_argument('--text_file', type=str, help='Text file for vocabulary (char LM)')
    parser.add_argument('--prompt', type=str, default='The', help='Text prompt for generation')
    parser.add_argument('--max_tokens', type=int, default=200, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling')
    
    # Demos
    parser.add_argument('--demo_recall', action='store_true', help='Run recall demo')
    parser.add_argument('--demo_gate', action='store_true', help='Visualize gate behavior')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Load model
    model, config = load_model(args.ckpt, args.device)
    
    # Recall demo
    if args.demo_recall:
        demo_recall(model, config.vocab_size, device=args.device)
    
    # Gate visualization
    if args.demo_gate:
        visualize_gate_behavior(model, device=args.device)
    
    # Text generation
    if args.text_file and not (args.demo_recall or args.demo_gate):
        # Load vocabulary from text file
        with open(args.text_file, 'r') as f:
            text = f.read()
        
        chars = sorted(list(set(text)))
        char2idx = {ch: i for i, ch in enumerate(chars)}
        idx2char = {i: ch for i, ch in enumerate(chars)}
        
        # Generate
        generated = generate_text(
            model, args.prompt, char2idx, idx2char,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device
        )
        
        print("="*60)
        print("GENERATED TEXT")
        print("="*60)
        print(generated)
        print("="*60)


if __name__ == '__main__':
    main()
