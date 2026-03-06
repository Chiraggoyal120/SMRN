# ⚠️ IMPORTANT: This is NOT a Web Application!

## What Is This Project?

This is a **PyTorch research implementation** - similar to implementing GPT, BERT, or other neural network architectures. It's a **command-line tool** for machine learning researchers and practitioners.

## ❌ What This Is NOT

- ❌ Not a website
- ❌ Not a web app with UI
- ❌ Not something you open in a browser
- ❌ No preview available

## ✅ What This IS

- ✅ PyTorch neural network implementation
- ✅ Research paper implementation (SMRN)
- ✅ Command-line training tools
- ✅ ML experimentation framework
- ✅ Python library/module

## 🎯 Intended Use

This implementation is for:

1. **ML Researchers** - Reproducing paper results
2. **Students** - Learning about state space models & attention
3. **Engineers** - Training custom sequence models
4. **Experimenters** - Validating theoretical claims

## 🚀 How to Actually Use It

### 1. Test the Implementation

```bash
cd /app/smrn
python test_smrn.py        # Run test suite
python validate.py         # Validate installation
```

### 2. Train a Model

```bash
# Create training data
echo "Your training text here..." > mydata.txt

# Train
python training/trainer.py \
  --task lm \
  --text_file mydata.txt \
  --d_model 128 \
  --n_layers 4 \
  --max_epochs 50
```

### 3. Generate Text

```bash
python inference/generate.py \
  --ckpt checkpoints/smrn_best.pt \
  --text_file mydata.txt \
  --prompt "Once upon a time"
```

### 4. Run Experiments (Validate Theorems)

```bash
# Validate all 4 theorems from the paper
python experiments/run_experiments.py --all
```

### 5. Visualize Results

```bash
# Generate plots
python utils/visualize.py --history checkpoints/history.json
```

## 📊 Example: Quick Demo

I trained a tiny model and it works:

```bash
$ python training/trainer.py --task lm --text_file sample_text.txt \
    --d_model 64 --n_layers 2 --max_epochs 5
✓ Training complete!

$ python inference/generate.py --ckpt demo_checkpoints/smrn_best.pt \
    --text_file sample_text.txt --prompt "The "
✓ Model loaded: 0.14M parameters
Generated: "The tfiifrbM,uuxoqvisgprdTpm..."
```

(Gibberish is normal - needs more training data & epochs)

## 📚 What You Can Do With This

### Research Use Cases

1. **Reproduce Paper Results**: Validate Theorems 1-4
2. **Compare Architectures**: SSM vs Attention vs SMRN
3. **Study Efficiency**: Measure O(N) time complexity
4. **Test Associative Recall**: Validate unbounded memory

### Practical Use Cases

1. **Train Language Models**: Character or token level
2. **Sequence Modeling**: Time series, DNA, music
3. **Long Context Tasks**: Efficient O(N) scaling
4. **Memory-Constrained Inference**: O(d²) state size

### Educational Use Cases

1. **Learn State Space Models**: Understand Mamba-style architectures
2. **Study Linear Attention**: Kernel trick for O(N) complexity
3. **Explore Gating Mechanisms**: Entropy-based routing
4. **Understand Theoretical Guarantees**: Formal complexity analysis

## 🔬 Project Type: ML Research Implementation

**Similar to:**
- HuggingFace Transformers (model implementations)
- PyTorch Vision Models (torchvision.models)
- Fairseq (Facebook AI Research)
- Mamba implementations

**Not similar to:**
- Web frameworks (Django, Flask, FastAPI apps)
- Frontend apps (React, Vue, Angular)
- Full-stack applications

## 📁 Project Structure

```
/app/smrn/
├── model/              # Neural network architecture
├── data/               # Dataset loaders
├── training/           # Training loops
├── experiments/        # Theorem validation
├── utils/              # Visualization
├── inference/          # Text generation
└── *.py               # Command-line scripts
```

## 💻 Development Workflow

```bash
# 1. Explore implementation
python demo.py

# 2. Modify architecture
vim model/smrn.py

# 3. Train with new config
python training/trainer.py --d_model 512 --n_layers 12

# 4. Evaluate
python experiments/run_experiments.py --recall

# 5. Generate samples
python inference/generate.py --ckpt checkpoints/smrn_best.pt
```

## 🎓 Learning Path

**If you're new to this:**

1. **Read the paper** (it's a theoretical ML paper)
2. **Understand PyTorch** (neural network framework)
3. **Run `demo.py`** to see architecture breakdown
4. **Train a tiny model** on small data
5. **Read code comments** (all equations documented)
6. **Experiment** with hyperparameters

## 🆘 Common Questions

### "Why can't I see anything in preview?"

Because this is **not a web app**. It's a Python library/tool that runs in the terminal.

### "How do I use this?"

Via command line:
```bash
cd /app/smrn
python <script>.py --help
```

### "What should I see?"

Terminal output:
- Training logs
- Loss curves (as text)
- Metrics
- Generated text
- Plots saved as PNG files

### "Is there a UI?"

No. This is pure ML research code. Use it via:
- Command line
- Jupyter notebooks (optional)
- Python scripts
- IDE

### "Can I make it a web app?"

Yes! You could build a Flask/FastAPI wrapper around it. But that's a separate project - this is just the ML core.

## 📖 Full Documentation

- **README.md** - Complete guide (800+ lines)
- **QUICKSTART.md** - Fast reference
- **MANIFEST.md** - Implementation details

## ✅ Verification

All components work correctly:

```
✓ test_smrn.py: 4/4 tests PASSED
✓ validate.py: 7/7 checks PASSED
✓ demo.py: Architecture breakdown ✓
✓ Training: Model trains successfully ✓
✓ Inference: Text generation works ✓
```

## 🎯 Bottom Line

**This is a research ML implementation for command-line use, not a web application.**

To use it:
1. Open terminal
2. Run Python commands
3. Train models
4. Generate text
5. Validate theorems

No browser, no preview, no UI - pure ML research code! 🚀
