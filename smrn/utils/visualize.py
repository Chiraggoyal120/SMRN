"""Visualization utilities for SMRN

All plots use matplotlib with Agg backend (headless mode)

Plots:
1. plot_loss_curves() - Training/validation curves
2. plot_gate_heatmap() - Gate activation per layer
3. plot_complexity() - Time vs N analysis
4. plot_gradient_norms() - Gradient stability over time
5. plot_ablation() - Ablation study results
6. plot_architecture() - SMRN architecture diagram (Figure 1)
7. generate_all_plots() - Generate all visualizations
"""

import matplotlib
matplotlib.use('Agg')  # Headless mode
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import json
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


def plot_loss_curves(history_path: str, save_path: str = 'plots/loss_curves.png'):
    """Plot training and validation loss/accuracy curves
    
    Args:
        history_path: Path to history.json
        save_path: Where to save the plot
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curves', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved loss curves to {save_path}")


def plot_gate_heatmap(model, save_path: str = 'plots/gate_heatmap.png', device='cpu'):
    """Plot gate activation heatmap per layer
    
    Args:
        model: SMRN model
        save_path: Where to save the plot
        device: Device for computation
    """
    model = model.to(device)
    model.eval()
    
    # Generate dummy input
    seq_len = 128
    vocab_size = model.config.vocab_size
    x = torch.randint(0, vocab_size, (1, seq_len), device=device)
    
    # Forward with gate values
    with torch.no_grad():
        _, gate_values = model(x, return_gate_values=True)
    
    # Extract mean gate values per layer
    gate_array = np.array([
        g.mean(dim=-1).cpu().numpy().squeeze() for g in gate_values
    ])  # (n_layers, seq_len)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(gate_array, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    
    ax.set_xlabel('Sequence Position', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Gate Activation Heatmap (Green=SSM, Red=Attention)', 
                 fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Gate Value (g)', fontsize=12)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved gate heatmap to {save_path}")


def plot_complexity(results: list, save_path: str = 'plots/complexity.png'):
    """Plot time complexity analysis
    
    Args:
        results: List of dicts with 'seq_len' and 'time_ms'
        save_path: Where to save the plot
    """
    seq_lens = [r['seq_len'] for r in results]
    times = [r['time_ms'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Linear plot
    axes[0].plot(seq_lens, times, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Sequence Length (N)', fontsize=12)
    axes[0].set_ylabel('Time (ms)', fontsize=12)
    axes[0].set_title('Time Complexity: Linear Scale', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Log-log plot (should be linear for O(N))
    axes[1].loglog(seq_lens, times, 'o-', linewidth=2, markersize=8, label='SMRN')
    
    # Add reference O(N) line
    x_ref = np.array(seq_lens)
    y_ref = times[0] * (x_ref / x_ref[0])  # Perfect O(N) scaling
    axes[1].loglog(x_ref, y_ref, '--', linewidth=2, alpha=0.5, label='O(N) reference')
    
    axes[1].set_xlabel('Sequence Length (N)', fontsize=12)
    axes[1].set_ylabel('Time (ms)', fontsize=12)
    axes[1].set_title('Time Complexity: Log-Log Scale', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved complexity plot to {save_path}")


def plot_gradient_norms(norms: list, save_path: str = 'plots/gradient_norms.png'):
    """Plot gradient norms over training steps
    
    Args:
        norms: List of gradient norms
        save_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = list(range(len(norms)))
    
    # Raw gradient norms
    ax.plot(steps, norms, alpha=0.3, label='Raw gradient norm')
    
    # Moving average
    window = 20
    if len(norms) >= window:
        moving_avg = np.convolve(norms, np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], moving_avg, linewidth=2, label=f'Moving avg (window={window})')
    
    # Max grad norm line (Theorem 4 enforcement)
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Max grad norm (clipping threshold)')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Stability (Theorem 4)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved gradient norms plot to {save_path}")


def plot_ablation(results: dict, save_path: str = 'plots/ablation.png'):
    """Plot ablation study results
    
    Args:
        results: Dict with variant names as keys, {'final_loss', 'n_params'} as values
        save_path: Where to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    variants = list(results.keys())
    losses = [results[v]['final_loss'] for v in variants]
    params = [results[v]['n_params'] / 1e6 for v in variants]  # In millions
    
    # Loss comparison
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    axes[0].bar(variants, losses, color=colors, alpha=0.8)
    axes[0].set_ylabel('Final Loss', fontsize=12)
    axes[0].set_title('Ablation: Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Parameter count comparison
    axes[1].bar(variants, params, color=colors, alpha=0.8)
    axes[1].set_ylabel('Parameters (Millions)', fontsize=12)
    axes[1].set_title('Ablation: Parameter Count', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved ablation plot to {save_path}")


def plot_architecture(save_path: str = 'plots/architecture.png'):
    """Plot SMRN architecture diagram (Figure 1 from paper)
    
    Shows: Input → [SSM Pathway | Linear Attention Pathway] → Entropy Gate → Output
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'SMRN Architecture', ha='center', fontsize=18, fontweight='bold')
    
    # Input
    input_box = FancyBboxPatch((0.5, 6), 1.5, 0.8, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 6.4, 'Input\nx_t', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # SSM Pathway (Pathway A - Compression)
    ssm_box = FancyBboxPatch((3, 6.5), 2, 1.2, 
                              boxstyle="round,pad=0.1", 
                              edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
    ax.add_patch(ssm_box)
    ax.text(4, 7.3, 'Selective SSM', ha='center', fontsize=11, fontweight='bold')
    ax.text(4, 7.0, '(Pathway A)', ha='center', fontsize=9, style='italic')
    ax.text(4, 6.7, 'h_t = Ã_t h_{t-1} + B̃_t x_t', ha='center', fontsize=8, family='monospace')
    
    # Linear Attention Pathway (Pathway B - Recall)
    attn_box = FancyBboxPatch((3, 4.8), 2, 1.2, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='darkblue', facecolor='lightcyan', linewidth=2)
    ax.add_patch(attn_box)
    ax.text(4, 5.6, 'Linear Attention', ha='center', fontsize=11, fontweight='bold')
    ax.text(4, 5.3, '(Pathway B)', ha='center', fontsize=9, style='italic')
    ax.text(4, 5.0, 'γ_t = φ(Q_t)S_t / φ(Q_t)Z_t', ha='center', fontsize=8, family='monospace')
    
    # Entropy Gate
    gate_box = FancyBboxPatch((6.5, 5.5), 2, 1.5, 
                               boxstyle="round,pad=0.1", 
                               edgecolor='darkred', facecolor='lightyellow', linewidth=2)
    ax.add_patch(gate_box)
    ax.text(7.5, 6.7, 'Entropy Gate', ha='center', fontsize=11, fontweight='bold')
    ax.text(7.5, 6.3, 'H_t = -Σ p log₂(p)', ha='center', fontsize=8, family='monospace')
    ax.text(7.5, 6.0, 'g_t = σ(W_g[y_ssm;y_attn;H_t])', ha='center', fontsize=7, family='monospace')
    ax.text(7.5, 5.7, 'y = g⊙y_ssm + (1-g)⊙y_attn', ha='center', fontsize=8, family='monospace')
    
    # Output
    output_box = FancyBboxPatch((9, 6), 0.8, 0.8, 
                                 boxstyle="round,pad=0.1", 
                                 edgecolor='black', facecolor='lightcoral', linewidth=2)
    ax.add_patch(output_box)
    ax.text(9.4, 6.4, 'Output', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    # Input to SSM
    arrow1 = FancyArrowPatch((2, 6.6), (3, 7.1), 
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow1)
    
    # Input to Attention
    arrow2 = FancyArrowPatch((2, 6.2), (3, 5.4), 
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow2)
    
    # SSM to Gate
    arrow3 = FancyArrowPatch((5, 7.1), (6.5, 6.7), 
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='darkgreen')
    ax.add_patch(arrow3)
    
    # Attention to Gate
    arrow4 = FancyArrowPatch((5, 5.4), (6.5, 6.0), 
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='darkblue')
    ax.add_patch(arrow4)
    
    # Gate to Output
    arrow5 = FancyArrowPatch((8.5, 6.4), (9, 6.4), 
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow5)
    
    # Complexity annotations
    ax.text(5, 3.5, 'Complexity Analysis', fontsize=13, fontweight='bold')
    ax.text(5, 3.1, '• Time: O(N) per sequence (Theorem 1 & 3)', fontsize=10)
    ax.text(5, 2.8, '• Memory: O(d²) independent of N (Theorem 3)', fontsize=10)
    ax.text(5, 2.5, '• Gradient: Bounded by |Ã_t| ≤ 1 (Theorem 4)', fontsize=10)
    ax.text(5, 2.2, '• Recall: O(dk·dv) associations with RFF (Theorem 2)', fontsize=10)
    
    # Legend
    ax.text(5, 1.5, 'Key Features', fontsize=12, fontweight='bold')
    ax.text(5, 1.2, '→ Selective: Ã_t and B̃_t depend on input (Mamba-style)', fontsize=9)
    ax.text(5, 0.9, '→ Dynamic: Gate adapts based on contextual entropy', fontsize=9)
    ax.text(5, 0.6, '→ Efficient: Linear time, constant memory', fontsize=9)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved architecture diagram to {save_path}")


def generate_all_plots(history_path: str = None, model=None, 
                       complexity_results: list = None, 
                       gradient_norms: list = None,
                       ablation_results: dict = None,
                       output_dir: str = 'plots'):
    """Generate all plots
    
    Args:
        history_path: Path to training history JSON
        model: SMRN model for gate heatmap
        complexity_results: Results from bench_time_complexity()
        gradient_norms: Results from bench_gradient_stability()
        ablation_results: Results from run_ablation()
        output_dir: Directory to save plots
    """
    print("\nGenerating all plots...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Architecture diagram (always generate)
    plot_architecture(f'{output_dir}/architecture.png')
    
    # Training curves
    if history_path and Path(history_path).exists():
        plot_loss_curves(history_path, f'{output_dir}/loss_curves.png')
    
    # Gate heatmap
    if model is not None:
        plot_gate_heatmap(model, f'{output_dir}/gate_heatmap.png')
    
    # Complexity
    if complexity_results:
        plot_complexity(complexity_results, f'{output_dir}/complexity.png')
    
    # Gradient norms
    if gradient_norms:
        plot_gradient_norms(gradient_norms, f'{output_dir}/gradient_norms.png')
    
    # Ablation
    if ablation_results:
        plot_ablation(ablation_results, f'{output_dir}/ablation.png')
    
    print(f"\n✓ All plots saved to {output_dir}/")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate SMRN visualizations')
    parser.add_argument('--history', type=str, help='Path to history.json')
    parser.add_argument('--output_dir', type=str, default='plots')
    
    args = parser.parse_args()
    
    # Generate architecture diagram (always)
    generate_all_plots(
        history_path=args.history,
        output_dir=args.output_dir
    )
