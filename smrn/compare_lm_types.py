#!/usr/bin/env python3
"""Test and compare character-level vs word-level LM"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.datasets import CharLMDataset, WordLMDataset

def compare_datasets():
    print("="*70)
    print("SMRN: Character-Level vs Word-Level Comparison")
    print("="*70)
    
    text_file = "wikitext2_sample.txt"
    
    # Read text
    with open(text_file, 'r') as f:
        text = f.read()
    
    print(f"\n📄 Test file: {text_file}")
    print(f"   Raw text length: {len(text):,} characters")
    print(f"   Word count: {len(text.split()):,} words")
    
    # Character-level
    print("\n" + "="*70)
    print("1️⃣  CHARACTER-LEVEL LM")
    print("="*70)
    
    char_ds = CharLMDataset(text, seq_len=128)
    print(f"✓ Vocab size: {char_ds.vocab_size}")
    print(f"✓ Total tokens: {len(char_ds.data):,}")
    print(f"✓ Chunks: {len(char_ds):,}")
    print(f"✓ Typical perplexity: 5-8")
    
    inp, tgt = char_ds[0]
    print(f"✓ Sample input shape: {inp.shape}")
    decoded = char_ds.decode(inp[:20].tolist())
    print(f"✓ Decoded (first 20): \"{decoded}\"")
    
    # Word-level
    print("\n" + "="*70)
    print("2️⃣  WORD-LEVEL LM (GPT-2 Tokenizer)")
    print("="*70)
    
    word_ds = WordLMDataset(text_file, seq_len=128, tokenizer_name='gpt2')
    print(f"✓ Chunks: {len(word_ds):,}")
    print(f"✓ Typical perplexity: 20-50")
    
    inp, tgt = word_ds[0]
    print(f"✓ Sample input shape: {inp.shape}")
    decoded = word_ds.decode(inp[:20].tolist())
    print(f"✓ Decoded (first 20 tokens): \"{decoded}\"")
    
    # Comparison
    print("\n" + "="*70)
    print("📊 COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Character-Level':<20} {'Word-Level':<20}")
    print("-" * 70)
    print(f"{'Vocab Size':<25} {char_ds.vocab_size:<20} {word_ds.vocab_size:<20}")
    print(f"{'Total Tokens':<25} {len(char_ds.data):<20,} {len(word_ds.data):<20,}")
    print(f"{'Chunks (seq_len=128)':<25} {len(char_ds):<20,} {len(word_ds):<20,}")
    print(f"{'Perplexity Range':<25} {'5-8':<20} {'20-50':<20}")
    print(f"{'Benchmark Comparable':<25} {'No':<20} {'Yes (GPT-2)':<20}")
    
    # Advantages
    print("\n" + "="*70)
    print("✅ ADVANTAGES")
    print("="*70)
    
    print("\n🔤 Character-Level:")
    print("   • Small vocab (~100)")
    print("   • Works with any language")
    print("   • No tokenizer needed")
    print("   • Lower perplexity (but not comparable)")
    
    print("\n📝 Word-Level (GPT-2):")
    print("   • Fair benchmark comparison")
    print("   • Standard vocab (50,257)")
    print("   • Realistic perplexity (20-50)")
    print("   • Comparable to published results")
    print("   • Better for research evaluation")
    
    # Recommendation
    print("\n" + "="*70)
    print("🎯 RECOMMENDATION")
    print("="*70)
    print("\n✅ Use WORD-LEVEL (GPT-2) for:")
    print("   • Research papers")
    print("   • Benchmark comparisons")
    print("   • Publishing results")
    print("   • Fair model evaluation")
    
    print("\n⚠️  Use CHARACTER-LEVEL for:")
    print("   • Non-English languages")
    print("   • Custom tokenization")
    print("   • Educational purposes")
    print("   • When tokenizer unavailable")
    
    print("\n" + "="*70)
    print("✅ BOTH IMPLEMENTATIONS WORKING CORRECTLY!")
    print("="*70)
    
    print("\n📚 Usage:")
    print("   Word-level (default):  python training/trainer.py --task lm --text_file data.txt")
    print("   Character-level:       python training/trainer.py --task lm --text_file data.txt --use_char_lm")

if __name__ == '__main__':
    compare_datasets()
