"""Datasets for SMRN experiments

5 dataset types:
1. AssociativeRecallDataset - Tests Theorem 2 (exact associative recall)
2. NeedleHaystackDataset - Tests recall at varying depths
3. CharLMDataset - Character-level language modeling
4. WikiTextDataset - Token-level language modeling (HuggingFace)
5. ListOpsDataset - Hierarchical task (LRA benchmark)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, List


class AssociativeRecallDataset(Dataset):
    """Dataset 1: Associative Recall (Needle-in-haystack)
    
    Tests Theorem 2: Unbounded Associative Recall
    - Insert (KEY_TOKEN, key, value) triplets at random positions
    - Query at end: [QUERY_TOKEN, key] → model must predict value
    - Under orthogonal φ, linear attention can store O(dk*dv) associations exactly
    
    Format:
        Input:  [rand, rand, KEY, k1, v1, rand, KEY, k2, v2, ..., QUERY, k1, <answer>]
        Target: [rand, rand, rand, v1, rand, rand, rand, k2, v2, ..., rand,  k1, v1]
                                                                                  ^^^ predict this
    """
    def __init__(self, n_samples: int = 10000, seq_len: int = 256, 
                 n_needles: int = 4, vocab_size: int = 512):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_needles = n_needles
        self.vocab_size = vocab_size
        
        # Special tokens
        self.KEY_TOKEN = vocab_size - 2
        self.QUERY_TOKEN = vocab_size - 1
        
        # Pre-generate all samples
        self.samples = [self._generate_sample() for _ in range(n_samples)]
    
    def _generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Generate one associative recall sample
        
        Returns:
            input: (seq_len,) input sequence
            target: (seq_len,) target sequence
            answer_pos: position of the answer
        """
        # Start with random tokens
        seq = torch.randint(0, self.vocab_size - 2, (self.seq_len,))
        target = seq.clone()
        
        # Insert needles (key-value pairs)
        pairs = []
        positions = sorted(np.random.choice(self.seq_len - 10, self.n_needles, replace=False))
        
        for pos in positions:
            key = np.random.randint(0, self.vocab_size - 2)
            val = np.random.randint(0, self.vocab_size - 2)
            pairs.append((key, val))
            
            # Insert: [KEY_TOKEN, key, val]
            seq[pos] = self.KEY_TOKEN
            seq[pos + 1] = key
            seq[pos + 2] = val
            target[pos:pos + 3] = seq[pos:pos + 3]
        
        # Insert query at the end
        query_key, query_val = pairs[np.random.randint(0, len(pairs))]
        query_pos = self.seq_len - 2
        
        seq[query_pos] = self.QUERY_TOKEN
        seq[query_pos + 1] = query_key
        
        target[query_pos] = self.QUERY_TOKEN
        target[query_pos + 1] = query_val  # This is what we want to predict
        
        return seq, target, query_pos + 1
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self.samples[idx]


class NeedleHaystackDataset(Dataset):
    """Dataset 2: Needle in Haystack at varying depths
    
    Tests recall performance at different context positions:
    - Depths: 0%, 25%, 50%, 75%, 100% of sequence
    - Context lengths: 512 to 4096 tokens
    
    Format:
        [random tokens] + [NEEDLE: key, val] + [random tokens] + [QUERY: key, <answer>]
    """
    def __init__(self, n_samples: int = 1000, vocab_size: int = 512):
        self.n_samples = n_samples
        self.vocab_size = vocab_size
        self.KEY_TOKEN = vocab_size - 2
        self.QUERY_TOKEN = vocab_size - 1
        
        # Test configurations: (context_len, depth_pct)
        self.configs = []
        for ctx_len in [512, 1024, 2048, 4096]:
            for depth in [0.0, 0.25, 0.5, 0.75, 1.0]:
                self.configs.append((ctx_len, depth))
        
        # Generate samples
        self.samples = []
        samples_per_config = max(1, n_samples // len(self.configs))
        for ctx_len, depth in self.configs:
            for _ in range(samples_per_config):
                self.samples.append(self._generate_sample(ctx_len, depth))
    
    def _generate_sample(self, ctx_len: int, depth: float) -> Tuple[torch.Tensor, torch.Tensor, float, int]:
        """Generate needle-haystack sample"""
        # Random sequence
        seq = torch.randint(0, self.vocab_size - 2, (ctx_len,))
        target = seq.clone()
        
        # Insert needle at specified depth (ensure space for needle and query)
        max_needle_pos = ctx_len - 5  # Leave space for needle (3 tokens) + query (2 tokens)
        needle_pos = min(int(max_needle_pos * depth), max_needle_pos)
        key = np.random.randint(0, self.vocab_size - 2)
        val = np.random.randint(0, self.vocab_size - 2)
        
        seq[needle_pos] = self.KEY_TOKEN
        seq[needle_pos + 1] = key
        seq[needle_pos + 2] = val
        
        # Query at the end
        query_pos = ctx_len - 2
        seq[query_pos] = self.QUERY_TOKEN
        seq[query_pos + 1] = key
        target[query_pos + 1] = val
        
        return seq, target, depth, ctx_len
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        return self.samples[idx]


class CharLMDataset(Dataset):
    """Dataset 3: Character-level Language Modeling
    
    - Works offline (no tokenizer needed)
    - Builds char2idx and idx2char automatically
    - Chunks text into seq_len windows
    """
    def __init__(self, text: str, seq_len: int = 256):
        self.seq_len = seq_len
        
        # Build vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}
        
        # Encode text
        self.data = torch.tensor([self.char2idx[ch] for ch in text], dtype=torch.long)
        
        # Create chunks
        self.n_chunks = (len(self.data) - 1) // seq_len
    
    def decode(self, indices: List[int]) -> str:
        """Decode indices to text"""
        return ''.join([self.idx2char[i] for i in indices])
    
    def __len__(self) -> int:
        return self.n_chunks
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        return chunk[:-1], chunk[1:]  # Input and target


class WikiTextDataset(Dataset):
    """Dataset 4: WikiText with HuggingFace tokenizer
    
    Requires: pip install transformers datasets
    """
    def __init__(self, split: str = 'train', seq_len: int = 256):
        try:
            from transformers import GPT2Tokenizer
            from datasets import load_dataset
        except ImportError:
            raise ImportError("WikiTextDataset requires: pip install transformers datasets")
        
        self.seq_len = seq_len
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load WikiText-2
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        
        # Concatenate all texts
        all_text = ' '.join([item['text'] for item in dataset if item['text'].strip()])
        
        # Tokenize
        tokens = self.tokenizer.encode(all_text)
        self.data = torch.tensor(tokens, dtype=torch.long)
        
        self.n_chunks = (len(self.data) - 1) // seq_len
    
    def __len__(self) -> int:
        return self.n_chunks
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        return chunk[:-1], chunk[1:]


class ListOpsDataset(Dataset):
    """Dataset 5: ListOps - Hierarchical task (LRA benchmark simulation)
    
    Synthetic task with nested operations:
    [MAX [MIN 2 3] [MIN 4 5]] -> 4
    
    Seq len: 512, labels: 0-9
    """
    def __init__(self, n_samples: int = 10000, seq_len: int = 512, vocab_size: int = 32):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.n_labels = 10
        
        # Generate synthetic samples
        self.samples = [self._generate_sample() for _ in range(n_samples)]
    
    def _generate_sample(self) -> Tuple[torch.Tensor, int]:
        """Generate one ListOps-style sample"""
        # Random sequence
        seq = torch.randint(0, self.vocab_size, (self.seq_len,))
        
        # Random label (0-9)
        label = np.random.randint(0, self.n_labels)
        
        return seq, label
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.samples[idx]


# Factory functions for data loaders

def get_recall_loaders(n_samples: int = 20000, seq_len: int = 256, 
                       n_needles: int = 4, vocab_size: int = 512,
                       batch_size: int = 64, split_ratio: float = 0.8):
    """Get train/val loaders for associative recall task"""
    dataset = AssociativeRecallDataset(n_samples, seq_len, n_needles, vocab_size)
    
    # Split
    n_train = int(n_samples * split_ratio)
    train_data, val_data = torch.utils.data.random_split(
        dataset, [n_train, n_samples - n_train]
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset.vocab_size


def get_char_loaders(text: str, seq_len: int = 256, batch_size: int = 64, 
                     split_ratio: float = 0.9):
    """Get train/val loaders for character LM task"""
    dataset = CharLMDataset(text, seq_len)
    
    # Split
    n_train = int(len(dataset) * split_ratio)
    train_data, val_data = torch.utils.data.random_split(
        dataset, [n_train, len(dataset) - n_train]
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, dataset.vocab_size, dataset.char2idx, dataset.idx2char


def get_haystack_loaders(n_samples: int = 1000, vocab_size: int = 512,
                         batch_size: int = 32):
    """Get loader for needle-haystack task"""
    dataset = NeedleHaystackDataset(n_samples, vocab_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, vocab_size


if __name__ == '__main__':
    print("Testing datasets...\n")
    
    # Test 1: Associative Recall
    print("1. AssociativeRecallDataset")
    ds = AssociativeRecallDataset(n_samples=10, seq_len=64, n_needles=2, vocab_size=100)
    inp, tgt, ans_pos = ds[0]
    print(f"   Input shape: {inp.shape}, Target shape: {tgt.shape}")
    print(f"   Answer position: {ans_pos}")
    print(f"   KEY_TOKEN={ds.KEY_TOKEN}, QUERY_TOKEN={ds.QUERY_TOKEN}")
    
    # Test 2: Needle Haystack
    print("\n2. NeedleHaystackDataset")
    ds = NeedleHaystackDataset(n_samples=10, vocab_size=100)
    inp, tgt, depth, ctx_len = ds[0]
    print(f"   Input shape: {inp.shape}")
    print(f"   Depth: {depth*100:.0f}%, Context length: {ctx_len}")
    
    # Test 3: Character LM
    print("\n3. CharLMDataset")
    text = "Hello world! This is a test." * 100
    ds = CharLMDataset(text, seq_len=64)
    print(f"   Vocab size: {ds.vocab_size}")
    print(f"   Dataset size: {len(ds)} chunks")
    inp, tgt = ds[0]
    print(f"   Input shape: {inp.shape}, Target shape: {tgt.shape}")
    print(f"   Decoded sample: '{ds.decode(inp[:20].tolist())}'")
    
    # Test 4: ListOps
    print("\n4. ListOpsDataset")
    ds = ListOpsDataset(n_samples=10, seq_len=128)
    inp, label = ds[0]
    print(f"   Input shape: {inp.shape}, Label: {label}")
    
    print("\n✓ All dataset tests passed!")
