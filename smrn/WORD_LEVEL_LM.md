# Word-Level Language Modeling with GPT-2 Tokenizer

## Overview

SMRN now supports **word-level language modeling** using GPT-2 tokenizer for fair benchmark comparison with other language models.

## Key Changes

### 1. New WordLMDataset (data/datasets.py)

```python
from data.datasets import WordLMDataset, get_word_loaders

# Create dataset
dataset = WordLMDataset('mytext.txt', seq_len=256, tokenizer_name='gpt2')

# Get data loaders
train_loader, val_loader, vocab_size, tokenizer = get_word_loaders(
    text_file='mytext.txt',
    seq_len=256,
    batch_size=32
)
```

**Features:**
- Uses GPT-2 tokenizer (transformers library)
- Vocab size: 50,257 tokens
- Same chunking as CharLMDataset
- Includes decode() method

### 2. Updated Trainer (training/trainer.py)

**Default behavior: Word-level LM**
```bash
python training/trainer.py --task lm --text_file data.txt
# Uses GPT-2 tokenizer by default
```

**Character-level LM (legacy):**
```bash
python training/trainer.py --task lm --text_file data.txt --use_char_lm
# Uses character-level tokenization
```

**New arguments:**
- `--tokenizer` - HuggingFace tokenizer name (default: gpt2)
- `--use_char_lm` - Enable character-level LM

### 3. Updated SMRNConfig (model/smrn.py)

```python
@dataclass
class SMRNConfig:
    vocab_size: int = 50257  # GPT-2 vocab size (default)
    # ... other params
```

## Usage Examples

### Quick Test

```bash
cd /app/smrn

# Test with sample text (310 words)
python training/trainer.py \
  --task lm \
  --text_file wikitext2_sample.txt \
  --d_model 128 \
  --n_layers 2 \
  --batch_size 4 \
  --seq_len 32 \
  --max_epochs 3
```

**Output:**
```
WordLMDataset initialized:
  Vocab size: 50257
  Total tokens: 400
  Using word-level LM with gpt2 tokenizer

Val Perplexity: 49,000 (expected for untrained tiny model)
✓ Training complete
```

### Full Training

```bash
# Download WikiText-2 or use your own text file
python training/trainer.py \
  --task lm \
  --text_file wikitext2.txt \
  --d_model 256 \
  --n_layers 8 \
  --d_state 16 \
  --batch_size 32 \
  --seq_len 128 \
  --max_epochs 10 \
  --lr 3e-4 \
  --max_grad_norm 0.5 \
  --save_dir checkpoints_wordlevel
```

### Use Different Tokenizer

```bash
# Use GPT-2 medium tokenizer
python training/trainer.py \
  --task lm \
  --text_file data.txt \
  --tokenizer gpt2-medium

# Use BERT tokenizer
python training/trainer.py \
  --task lm \
  --text_file data.txt \
  --tokenizer bert-base-uncased
```

### Character-Level (Legacy)

```bash
python training/trainer.py \
  --task lm \
  --text_file data.txt \
  --use_char_lm
```

## Perplexity Comparison

### Character-Level
- Typical perplexity: **5-8** (lower vocab size ~100)
- Not directly comparable to word-level models

### Word-Level (GPT-2 Tokenizer)
- Typical perplexity: **20-50** for well-trained models
- **Directly comparable** to GPT-2, BERT, etc.
- Vocab size: 50,257

### Why Word-Level?

1. **Fair Benchmarking**: Standard for comparing language models
2. **Realistic Metrics**: Perplexity comparable to published results
3. **Better Evaluation**: Word-level matches how humans read
4. **Research Standard**: Papers report word-level perplexity

## Expected Results

### Small Model (untrained)
```
Epochs: 3
Perplexity: ~49,000 (random guessing over 50K vocab)
```

### Trained Model (after convergence)
```
Epochs: 50-100
Perplexity: 20-50 (depends on data size and model capacity)
```

For comparison:
- GPT-2 (117M): ~29 perplexity on WikiText-2
- LSTM baseline: ~100-150 perplexity
- Transformer-XL: ~24 perplexity

## Technical Details

### Tokenization

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokens = tokenizer.encode("Hello world!")
# [15496, 995, 0]

text = tokenizer.decode(tokens)
# "Hello world!"
```

### Vocabulary

- **Size**: 50,257 tokens
- **Includes**: 
  - Common words: "the", "a", "is"
  - Subwords: "ing", "tion"
  - Special tokens: `<|endoftext|>`

### Chunking

Same as CharLMDataset:
1. Tokenize entire text
2. Split into `seq_len` chunks
3. Input: chunk[:-1]
4. Target: chunk[1:]

## Installation

```bash
# Install transformers library
pip install transformers

# Already included in requirements.txt
```

## Testing

```bash
# Test WordLMDataset
cd /app/smrn
python -c "
from data.datasets import WordLMDataset
ds = WordLMDataset('wikitext2_sample.txt', seq_len=32)
print(f'✓ Vocab size: {ds.vocab_size}')
print(f'✓ Chunks: {len(ds)}')
"

# Test training
python training/trainer.py \
  --task lm \
  --text_file wikitext2_sample.txt \
  --d_model 64 \
  --n_layers 1 \
  --max_epochs 2
```

## Migration Guide

### From Character-Level to Word-Level

**Before (Character-level):**
```bash
python training/trainer.py --task lm --text_file data.txt
# Vocab ~100, Perplexity 5-8
```

**After (Word-level - default):**
```bash
python training/trainer.py --task lm --text_file data.txt
# Vocab 50257, Perplexity 20-50
```

**Keep Character-level:**
```bash
python training/trainer.py --task lm --text_file data.txt --use_char_lm
```

## Benchmark Commands

### Small Model (Quick Test)
```bash
python training/trainer.py \
  --task lm \
  --text_file wikitext2.txt \
  --d_model 256 \
  --n_layers 4 \
  --batch_size 64 \
  --seq_len 256 \
  --max_epochs 20
```

### Medium Model (Research)
```bash
python training/trainer.py \
  --task lm \
  --text_file wikitext2.txt \
  --d_model 512 \
  --n_layers 8 \
  --batch_size 32 \
  --seq_len 512 \
  --max_epochs 50
```

### Large Model (Full Benchmark)
```bash
python training/trainer.py \
  --task lm \
  --text_file wikitext2.txt \
  --d_model 768 \
  --n_layers 12 \
  --batch_size 16 \
  --seq_len 1024 \
  --max_epochs 100
```

## Files Modified

1. **data/datasets.py** - Added WordLMDataset + get_word_loaders()
2. **training/trainer.py** - Default to word-level, add --tokenizer, --use_char_lm
3. **model/smrn.py** - Default vocab_size = 50257
4. **data/__init__.py** - Export new classes

## Commit

```bash
git add -A
git commit -m "feat: word-level GPT-2 tokenizer for fair benchmark comparison"
git push origin main
```

## Summary

✅ **Word-level LM**: Default for fair benchmarking  
✅ **GPT-2 tokenizer**: Standard 50,257 vocab  
✅ **Backward compatible**: Use `--use_char_lm` for character-level  
✅ **Realistic perplexity**: 20-50 range for trained models  
✅ **Easy to use**: Same CLI, better defaults  

**Status: Production ready for benchmark comparison!** 🚀
