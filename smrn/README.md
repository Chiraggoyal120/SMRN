# SMRN: Selective Memory and Recall Network

**PyTorch Implementation of "Selective Memory and Recall Network: A Theoretical Analysis for Linear-Time Agents"**

*Authors: Chirag Goyal, Manoj Kumar — VIT Bhopal University*

---

## Overview

SMRN is a novel neural architecture that achieves **O(N) time complexity** and **O(d²) memory** (independent of sequence length) by combining:

1. **Selective SSM** (Pathway A - Compression): Mamba-style state space model with input-dependent discretization
2. **Linear Attention** (Pathway B - Recall): Kernel-based attention with O(N) complexity via recurrent formulation  
3. **Entropy Gate** (Dynamic Gating): Contextual entropy-based adaptive routing between pathways

### Key Features

✅ **Linear Time**: O(N) per sequence (Theorem 1 & 3)  
✅ **Constant Memory**: O(d²) independent of N (Theorem 3)  
✅ **Unbounded Recall**: O(dk·dv) associations with RFF (Theorem 2)  
✅ **Gradient Stability**: Bounded gradients via A matrix design (Theorem 4)  
✅ **Dynamic Adaptation**: Entropy-based gating adapts to context

---

## Installation

```bash
cd /app/smrn
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- einops, numpy, matplotlib, tqdm

---

## Quick Start

### 1. Test Architecture

```bash
python model/smrn.py
```

### 2. Test Datasets

```bash
python data/datasets.py
```

### 3. Train on Synthetic Recall Task

```bash
python training/trainer.py --task recall --d_model 256 --n_layers 6 --max_epochs 30
```

### 4. Train Character-Level Language Model

```bash
# Create a text file (e.g., input.txt)
echo "Your training text here..." > input.txt

python training/trainer.py \
  --task lm \
  --text_file input.txt \
  --d_model 256 \
  --n_layers 6 \
  --seq_len 256 \
  --max_epochs 50
```

### 5. Validate All Theorems

```bash
python experiments/run_experiments.py --all
```

### 6. Generate Text

```bash
python inference/generate.py \
  --ckpt checkpoints/smrn_best.pt \
  --text_file input.txt \
  --prompt "Once upon a time" \
  --max_tokens 200 \
  --temperature 0.8
```

---

## Architecture Components

### Component 1: Selective SSM (Pathway A)

**Continuous-time SSM:**
```
dh/dt = A(t)h(t) + B(t)x(t)  [State equation]
y(t) = C(t)h(t) + D(t)x(t)   [Output equation]
```

**Zero-Order Hold (ZOH) Discretization:**
```
h_t = Ã_t * h_{t-1} + B̃_t * x_t
where Ã_t = exp(Δ*A), B̃_t = Δ*B_t
```

**Key Properties:**
- Input-dependent (selective): Ã_t and B̃_t are functions of x_t
- A matrix log-parameterized for stability: ||Ã_t|| ≤ 1
- O(d²s) per step → O(N) total (Theorem 1)

### Component 2: Linear Attention (Pathway B)

**Standard attention O(N²):**
```
Attn(Q,K,V) = softmax(QK^T)V
```

**Reformulated as O(N) via kernel trick:**
```
φ(Q)(φ(K)^T V) / φ(Q)(φ(K)^T 1)
```

**Recurrent state form:**
```
S_t = S_{t-1} + φ(K_t)^T V_t     [Associative memory]
Z_t = Z_{t-1} + φ(K_t)           [Normalizer]
γ_t = φ(Q_t) S_t / φ(Q_t) Z_t   [Recall]
```

**Feature maps:**
- Simple: φ(x) = ELU(x) + 1
- RFF (Theorem 2): φ(x) = √(2/m) [cos(ωx), sin(ωx)], ω ~ N(0, σ²I)

### Component 3: Entropy Gate

**Contextual entropy:**
```
H_t = -Σ p(x_i) log₂(p(x_i)) over window w
```

**Gate computation:**
```
g_t = σ(W_g [y_ssm; y_attn; H_t] + b_g)
```

**Output:**
```
y_out = g_t ⊙ y_ssm + (1 - g_t) ⊙ y_attn
```

**Interpretation:**
- High entropy (surprising) → trust recall (attention)
- Low entropy (predictable) → trust compression (SSM)

---

## Experiments

### Experiment 1: Time Complexity (Theorem 1 & 3)

```bash
python experiments/run_experiments.py --time
```

Tests O(N) scaling on seq_lens = [64, 128, 256, 512, 1024, 2048].

**Expected output:** Time ratio ~2x when N doubles.

### Experiment 2: Memory Complexity (Theorem 3)

```bash
python experiments/run_experiments.py --memory
```

Compares SMRN's O(d²) state with Transformer's O(Nd) KV cache.

**Key insight:** SMRN state is CONSTANT regardless of sequence length.

### Experiment 3: Gradient Stability (Theorem 4)

```bash
python experiments/run_experiments.py --gradients
```

Tracks gradient norms over 300 training steps.

**Theorem 4 guarantees:**
- |∂h_t/∂h_{t-1}| = ||Ã_t|| ≤ 1 (A is negative)
- Gate values ∈ [0,1] (sigmoid bounded)

### Experiment 4: Associative Recall (Theorem 2)

```bash
python experiments/run_experiments.py --recall
```

Tests recall accuracy with varying n_needles and seq_lens.

**Theorem 2:** Under orthogonal φ (RFF), can store O(dk·dv) associations exactly.

### Experiment 5: Ablation Study

```bash
python experiments/run_experiments.py --ablation
```

Compares:
- SSM only
- Attention only  
- Full SMRN (SSM + Attention + Gate)

### Experiment 6: Needle in Haystack

```bash
python experiments/run_experiments.py --haystack
```

Tests recall at depths [0%, 25%, 50%, 75%, 100%] and context lengths [64, 128, 256].

---

## Visualization

### Generate All Plots

```bash
python utils/visualize.py --history checkpoints/history.json --output_dir plots
```

**Generated plots:**
1. `architecture.png` - SMRN architecture diagram (Figure 1)
2. `loss_curves.png` - Training/validation loss and accuracy
3. `gate_heatmap.png` - Gate activation per layer (heatmap)
4. `complexity.png` - Time vs N analysis (linear + log-log)
5. `gradient_norms.png` - Gradient stability over training
6. `ablation.png` - Ablation study comparison

---

## Inference & Demos

### 1. Text Generation

```bash
python inference/generate.py \
  --ckpt checkpoints/smrn_best.pt \
  --text_file input.txt \
  --prompt "To be or not to be" \
  --max_tokens 200 \
  --temperature 0.8 \
  --top_k 40 \
  --top_p 0.9
```

### 2. Associative Recall Demo

```bash
python inference/generate.py \
  --ckpt checkpoints/smrn_best.pt \
  --demo_recall
```

Runs 10 trials of key-value recall and reports accuracy.

### 3. Gate Behavior Visualization

```bash
python inference/generate.py \
  --ckpt checkpoints/smrn_best.pt \
  --demo_gate
```

Shows ASCII bar chart of gate values per layer:
- █ = SSM dominant (g > 0.7)
- ▓ = Balanced (0.4 < g < 0.7)
- ░ = Attention dominant (g < 0.4)

---

## Training Arguments

```bash
python training/trainer.py --help
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `recall` | Task: recall, lm, haystack, listops |
| `--d_model` | 256 | Model dimension |
| `--n_layers` | 6 | Number of SMRN blocks |
| `--d_state` | 16 | SSM state dimension |
| `--window_size` | 8 | Entropy window size |
| `--no_rff` | False | Disable Random Fourier Features |
| `--batch_size` | 64 | Batch size |
| `--seq_len` | 256 | Sequence length |
| `--max_epochs` | 50 | Maximum epochs |
| `--lr` | 3e-4 | Learning rate |
| `--max_grad_norm` | 1.0 | Gradient clipping (Theorem 4) |
| `--n_samples` | 20000 | Dataset size |
| `--text_file` | None | Text file for character LM |
| `--save_dir` | `checkpoints` | Save directory |
| `--resume` | None | Resume from checkpoint |

---

## Datasets

SMRN includes 5 dataset types:

### 1. AssociativeRecallDataset

**Purpose:** Test Theorem 2 (exact associative recall)

**Format:**
```
Input:  [rand, rand, KEY, k1, v1, rand, KEY, k2, v2, ..., QUERY, k1, <answer>]
Target: [rand, rand, rand, v1, rand, rand, rand, k2, v2, ..., rand,  k1, v1]
```

### 2. NeedleHaystackDataset

**Purpose:** Test recall at varying depths

**Configurations:**
- Depths: 0%, 25%, 50%, 75%, 100%
- Context lengths: 512, 1024, 2048, 4096

### 3. CharLMDataset

**Purpose:** Character-level language modeling

**Features:**
- No tokenizer needed (works offline)
- Builds char2idx/idx2char automatically
- Includes decode() method

### 4. WikiTextDataset

**Purpose:** Token-level language modeling

**Requirements:** `pip install transformers datasets`

### 5. ListOpsDataset

**Purpose:** Hierarchical task (LRA benchmark simulation)

**Specifications:**
- Seq len: 512
- Labels: 0-9

---

## Theoretical Guarantees

### Theorem 1: SelectiveSSM Linear Complexity

**Statement:** SelectiveSSM has O(N) time complexity per sequence.

**Proof sketch:**
- Each step: O(d²s) operations (constant)
- N steps: O(N × d²s) = O(N)

**Validation:** `python experiments/run_experiments.py --time`

### Theorem 2: Unbounded Associative Recall

**Statement:** Linear attention with orthogonal feature map φ (RFF) can store O(dk·dv) associations exactly.

**Key insight:**
- Recurrent state S_t accumulates key-value associations
- Under orthogonal φ, no interference between different keys
- Memory capacity scales with feature dimension

**Validation:** `python experiments/run_experiments.py --recall`

### Theorem 3: SMRN Complexity

**Statement:** Full SMRN has:
- Time: O(N) per sequence
- Memory: O(d²) independent of N

**Breakdown:**
- SSM state: d_model × d_state per layer
- Attention state: d_model × d_model × 2 per layer (S and Z)
- Total: O(d²) × n_layers (constant, independent of N)

**Comparison with Transformer:**
- Transformer KV cache: 2 × n_layers × N × d_model = O(Nd)
- SMRN: Independent of N

**Validation:** `python experiments/run_experiments.py --memory`

### Theorem 4: Gradient Stability

**Statement:** SMRN gradients remain bounded during training:
- |∂h_t/∂h_{t-1}| = ||Ã_t|| ≤ 1
- Gate outputs ∈ [0,1]

**Mechanisms:**
1. A matrix is log-parameterized and negative: A = -exp(A_log)
2. This ensures exp(Δ*A) has norm ≤ 1
3. Gate uses sigmoid: bounded in [0,1]
4. Gradient clipping to max_grad_norm enforces upper bound

**Validation:** `python experiments/run_experiments.py --gradients`

---

## File Structure

```
smrn/
├── model/
│   ├── __init__.py
│   └── smrn.py              # Core architecture (SSM, Attention, Gate, SMRN)
├── data/
│   ├── __init__.py
│   └── datasets.py          # 5 dataset types + factory functions
├── training/
│   ├── __init__.py
│   └── trainer.py           # Training engine (AdamW, cosine LR, AMP)
├── experiments/
│   ├── __init__.py
│   └── run_experiments.py   # Validate 4 theorems + ablations
├── utils/
│   ├── __init__.py
│   └── visualize.py         # Matplotlib plots (headless)
├── inference/
│   ├── __init__.py
│   └── generate.py          # Text generation + demos
├── requirements.txt
└── README.md
```

---

## Examples

### Example 1: Train and Validate

```bash
# Train on recall task
python training/trainer.py \
  --task recall \
  --d_model 256 \
  --n_layers 6 \
  --max_epochs 30 \
  --batch_size 64

# Validate all theorems
python experiments/run_experiments.py --all

# Generate plots
python utils/visualize.py --history checkpoints/history.json
```

### Example 2: Character Language Model

```bash
# Prepare text
wget https://www.gutenberg.org/files/1342/1342-0.txt -O pride_prejudice.txt

# Train
python training/trainer.py \
  --task lm \
  --text_file pride_prejudice.txt \
  --d_model 256 \
  --n_layers 8 \
  --seq_len 512 \
  --max_epochs 50

# Generate
python inference/generate.py \
  --ckpt checkpoints/smrn_best.pt \
  --text_file pride_prejudice.txt \
  --prompt "It is a truth universally acknowledged" \
  --max_tokens 500 \
  --temperature 0.7
```

### Example 3: Ablation Study

```bash
# Run ablation
python experiments/run_experiments.py --ablation

# Results will show:
# - SSM Only: Compression pathway only
# - Attention Only: Recall pathway only
# - Full SMRN: Dynamic gating between both
```

---

## Citation

If you use this implementation, please cite:

```bibtex
@article{goyal2024smrn,
  title={Selective Memory and Recall Network: A Theoretical Analysis for Linear-Time Agents},
  author={Goyal, Chirag and Kumar, Manoj},
  institution={VIT Bhopal University},
  year={2024}
}
```

---

## Key Insights

1. **Dual Pathways:** SMRN maintains both compression (SSM) and recall (attention) pathways, dynamically selecting between them based on context.

2. **O(N) Efficiency:** Unlike standard Transformers (O(N²)), SMRN achieves linear time complexity through:
   - Recurrent SSM formulation (O(N))
   - Kernel-based linear attention (O(N))

3. **Constant Memory:** State size is O(d²), independent of sequence length. Critical for long contexts.

4. **Entropy-Based Gating:** High entropy (surprising context) → use attention (recall). Low entropy (predictable) → use SSM (compression).

5. **Gradient Stability:** Log-parameterized A matrix ensures stable training via bounded gradients (Theorem 4).

6. **Orthogonal Features:** RFF provides orthogonal φ for exact associative recall (Theorem 2).

---

## Troubleshooting

### Issue: Out of memory during training

**Solution:** Reduce `--batch_size` or `--seq_len`

### Issue: Poor recall accuracy

**Solution:** 
- Enable RFF: Don't use `--no_rff`
- Increase model capacity: `--d_model 512 --n_layers 8`
- Train longer: `--max_epochs 100`

### Issue: Gradient explosion

**Solution:** Already handled by Theorem 4 implementation. Check that `--max_grad_norm 1.0` is set.

### Issue: Slow training

**Solution:**
- Enable mixed precision (default: `--use_amp`)
- Reduce sequence length: `--seq_len 128`
- Use GPU: Code auto-detects CUDA

---

## Future Work

- [ ] Multi-head attention variant
- [ ] Sparse attention integration
- [ ] Bidirectional SMRN
- [ ] Pretrained models on large corpora
- [ ] Integration with HuggingFace Transformers
- [ ] Longer context evaluation (8K, 16K, 32K tokens)
- [ ] Multimodal extensions

---

## License

MIT License - See implementation for details.

---

## Contact

For questions or issues, please open a GitHub issue or contact the paper authors.

---

**Built with ❤️ for efficient long-sequence modeling**
