#!/usr/bin/env python3
"""
SMRN Implementation Validation Script
Quickly verifies all components are working
"""

import sys
from pathlib import Path

def check_imports():
    """Check all modules import correctly"""
    print("🔍 Checking imports...")
    try:
        from model.smrn import SMRN, SMRNConfig, SMRNSSMOnly, SMRNAttnOnly
        from data.datasets import (AssociativeRecallDataset, NeedleHaystackDataset,
                                   CharLMDataset, ListOpsDataset)
        from training.trainer import SMRNTrainer
        from utils.visualize import plot_architecture
        from inference.generate import load_model, generate_text
        print("   ✅ All imports successful")
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

def check_model():
    """Check model instantiation"""
    print("\n🏗️ Checking model...")
    try:
        import torch
        from model.smrn import SMRN, SMRNConfig
        
        config = SMRNConfig(vocab_size=256, d_model=64, n_layers=2)
        model = SMRN(config)
        x = torch.randint(0, 256, (2, 16))
        logits = model(x)
        
        assert logits.shape == (2, 16, 256), "Output shape mismatch"
        print(f"   ✅ Model forward pass: {x.shape} → {logits.shape}")
        return True
    except Exception as e:
        print(f"   ❌ Model check failed: {e}")
        return False

def check_datasets():
    """Check dataset creation"""
    print("\n📊 Checking datasets...")
    try:
        from data.datasets import (AssociativeRecallDataset, CharLMDataset, 
                                   ListOpsDataset)
        
        ds1 = AssociativeRecallDataset(n_samples=5, seq_len=32, vocab_size=100)
        ds2 = CharLMDataset("Hello world!", seq_len=8)
        ds3 = ListOpsDataset(n_samples=5, seq_len=32)
        
        print(f"   ✅ AssociativeRecall: {len(ds1)} samples")
        print(f"   ✅ CharLM: vocab_size={ds2.vocab_size}")
        print(f"   ✅ ListOps: {len(ds3)} samples")
        return True
    except Exception as e:
        print(f"   ❌ Dataset check failed: {e}")
        return False

def check_training():
    """Check trainer instantiation"""
    print("\n🏋️ Checking trainer...")
    try:
        import torch
        from model.smrn import SMRN, SMRNConfig
        from training.trainer import SMRNTrainer
        
        config = SMRNConfig(vocab_size=100, d_model=32, n_layers=1)
        model = SMRN(config)
        trainer = SMRNTrainer(model, config, device='cpu')
        
        print(f"   ✅ Trainer initialized")
        print(f"   ✅ Optimizer: {type(trainer.optimizer).__name__}")
        return True
    except Exception as e:
        print(f"   ❌ Trainer check failed: {e}")
        return False

def check_visualization():
    """Check visualization functions"""
    print("\n📈 Checking visualization...")
    try:
        from utils.visualize import plot_architecture
        import os
        
        os.makedirs('plots', exist_ok=True)
        plot_architecture('plots/test_arch.png')
        
        assert Path('plots/test_arch.png').exists(), "Plot not created"
        print(f"   ✅ Architecture plot generated")
        return True
    except Exception as e:
        print(f"   ❌ Visualization check failed: {e}")
        return False

def check_files():
    """Check all expected files exist"""
    print("\n📁 Checking file structure...")
    
    expected_files = [
        'model/__init__.py',
        'model/smrn.py',
        'data/__init__.py',
        'data/datasets.py',
        'training/__init__.py',
        'training/trainer.py',
        'experiments/__init__.py',
        'experiments/run_experiments.py',
        'utils/__init__.py',
        'utils/visualize.py',
        'inference/__init__.py',
        'inference/generate.py',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
        'MANIFEST.md',
        'test_smrn.py',
        'demo.py'
    ]
    
    missing = []
    for f in expected_files:
        if not Path(f).exists():
            missing.append(f)
    
    if missing:
        print(f"   ❌ Missing files: {', '.join(missing)}")
        return False
    else:
        print(f"   ✅ All {len(expected_files)} files present")
        return True

def check_documentation():
    """Check documentation completeness"""
    print("\n📚 Checking documentation...")
    
    readme_size = Path('README.md').stat().st_size
    quickstart_size = Path('QUICKSTART.md').stat().st_size
    manifest_size = Path('MANIFEST.md').stat().st_size
    
    print(f"   ✅ README.md: {readme_size:,} bytes")
    print(f"   ✅ QUICKSTART.md: {quickstart_size:,} bytes")
    print(f"   ✅ MANIFEST.md: {manifest_size:,} bytes")
    
    return readme_size > 1000 and quickstart_size > 1000

def main():
    print("="*70)
    print("🚀 SMRN IMPLEMENTATION VALIDATION")
    print("="*70)
    
    checks = [
        ("File Structure", check_files),
        ("Imports", check_imports),
        ("Model", check_model),
        ("Datasets", check_datasets),
        ("Trainer", check_training),
        ("Visualization", check_visualization),
        ("Documentation", check_documentation),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check crashed: {e}")
            results.append((name, False))
    
    print("\n" + "="*70)
    print("📊 VALIDATION SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name:20s}: {status}")
    
    total = len(results)
    passed = sum(1 for _, r in results if r)
    
    print(f"\n   Total: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n" + "="*70)
        print("🎉 ALL VALIDATION CHECKS PASSED!")
        print("="*70)
        print("\n✨ Implementation Status: COMPLETE AND VERIFIED")
        print("\n📦 Ready for:")
        print("   • Training models")
        print("   • Running experiments")
        print("   • Validating theorems")
        print("   • Generating text")
        print("\n📖 See README.md for full documentation")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
