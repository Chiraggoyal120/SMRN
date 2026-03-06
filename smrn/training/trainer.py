"""Training engine for SMRN

Features:
- AdamW optimizer with separate decay/no-decay param groups
- Cosine learning rate schedule with warmup
- Mixed precision training (torch.cuda.amp)
- Gradient clipping to enforce Theorem 4
- Early stopping with patience
- Checkpointing and history tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import math
from typing import Dict, Tuple, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))

from model.smrn import SMRN, SMRNConfig, SMRNSSMOnly, SMRNAttnOnly
from data.datasets import (
    get_recall_loaders, get_char_loaders, get_haystack_loaders,
    AssociativeRecallDataset, ListOpsDataset
)


class SMRNTrainer:
    """Training engine for SMRN models
    
    Implements:
    - Theorem 4 enforcement: gradient clipping to max_grad_norm
    - Mixed precision for efficiency
    - Cosine LR schedule with warmup
    """
    def __init__(self, model: nn.Module, config: SMRNConfig, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer: separate decay/no-decay groups
        self.optimizer = self._configure_optimizer()
        
        # Learning rate scheduler
        self.scheduler = None  # Will be set after computing total steps
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp and device == 'cuda' else None
        
        # History tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
        # Save directory
        os.makedirs(config.save_dir, exist_ok=True)
    
    def _configure_optimizer(self) -> torch.optim.Optimizer:
        """Configure AdamW with weight decay only for weights (not biases/LayerNorm)"""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # No weight decay for biases and LayerNorm
            if 'bias' in name or 'norm' in name or 'embed' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.config.lr, betas=(0.9, 0.95))
        
        return optimizer
    
    def _get_lr_scheduler(self, total_steps: int):
        """Cosine learning rate schedule with warmup"""
        def lr_lambda(step: int) -> float:
            # Warmup
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            
            # Cosine decay
            progress = (step - self.config.warmup_steps) / (total_steps - self.config.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            # Scale between min_lr and lr
            lr_range = self.config.lr - self.config.min_lr
            return self.config.min_lr / self.config.lr + (lr_range / self.config.lr) * cosine_decay
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader: DataLoader, task: str = 'lm') -> Dict[str, float]:
        """Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            task: 'recall', 'lm', 'haystack', or 'listops'
        Returns:
            metrics: {'loss': float, 'acc': float, 'grad_norm': float}
        """
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        grad_norms = []
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            # Parse batch based on task
            if task in ['recall', 'haystack']:
                if len(batch) == 3:  # Associative recall
                    inputs, targets, ans_pos = batch
                else:  # Needle haystack
                    inputs, targets, _, _ = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            elif task == 'lm':
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            elif task == 'listops':
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                logits = self.model(inputs)
                
                if task == 'listops':
                    # Classification: use last token's logits
                    logits = logits[:, -1, :self.config.vocab_size]
                    loss = F.cross_entropy(logits, labels)
                    acc = (logits.argmax(dim=-1) == labels).float().mean()
                else:
                    # Language modeling: predict next token
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1)
                    )
                    
                    if task in ['recall', 'haystack']:
                        # Accuracy on answer position only
                        if len(batch) == 3:
                            # Use ans_pos
                            preds = logits.argmax(dim=-1)
                            correct = sum([
                                (preds[i, ans_pos[i]] == targets[i, ans_pos[i]]).item()
                                for i in range(len(ans_pos))
                            ])
                            acc = correct / len(ans_pos)
                        else:
                            # Last position for haystack
                            acc = (logits[:, -1, :].argmax(dim=-1) == targets[:, -1]).float().mean()
                    else:
                        # Overall accuracy
                        acc = (logits.argmax(dim=-1) == targets).float().mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()
            
            # Gradient clipping (Theorem 4: ensures gradient stability)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            grad_norms.append(grad_norm.item())
            
            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Accumulate metrics
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': acc.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        return {
            'loss': total_loss / total_samples,
            'acc': total_acc / total_samples,
            'grad_norm': sum(grad_norms) / len(grad_norms)
        }
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader, task: str = 'lm') -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        
        for batch in tqdm(val_loader, desc='Evaluating'):
            # Parse batch
            if task in ['recall', 'haystack']:
                if len(batch) == 3:
                    inputs, targets, ans_pos = batch
                else:
                    inputs, targets, _, _ = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            elif task == 'lm':
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            elif task == 'listops':
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(inputs)
            
            if task == 'listops':
                logits = logits[:, -1, :self.config.vocab_size]
                loss = F.cross_entropy(logits, labels)
                acc = (logits.argmax(dim=-1) == labels).float().mean()
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )
                
                if task in ['recall', 'haystack']:
                    if len(batch) == 3:
                        preds = logits.argmax(dim=-1)
                        correct = sum([
                            (preds[i, ans_pos[i]] == targets[i, ans_pos[i]]).item()
                            for i in range(len(ans_pos))
                        ])
                        acc = correct / len(ans_pos)
                    else:
                        acc = (logits[:, -1, :].argmax(dim=-1) == targets[:, -1]).float().mean()
                else:
                    acc = (logits.argmax(dim=-1) == targets).float().mean()
            
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size
            total_samples += batch_size
        
        metrics = {
            'loss': total_loss / total_samples,
            'acc': total_acc / total_samples
        }
        
        # Compute perplexity for LM tasks
        if task == 'lm':
            metrics['perplexity'] = math.exp(metrics['loss'])
        
        return metrics
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, task: str = 'lm'):
        """Full training loop with early stopping"""
        print(f"\nTraining SMRN for {self.config.max_epochs} epochs...")
        print(f"Task: {task}")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M\n")
        
        # Setup LR scheduler
        total_steps = len(train_loader) * self.config.max_epochs
        self.scheduler = self._get_lr_scheduler(total_steps)
        
        for epoch in range(self.config.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, task)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader, task)
            
            # Log
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['acc']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.4f}")
            if 'perplexity' in val_metrics:
                print(f"Val Perplexity: {val_metrics['perplexity']:.2f}")
            print(f"Grad Norm: {train_metrics['grad_norm']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.epochs_no_improve = 0
                self.save('best')
                print("✓ New best model saved!")
            else:
                self.epochs_no_improve += 1
            
            # Early stopping
            if self.epochs_no_improve >= self.config.patience:
                print(f"\nEarly stopping after {epoch + 1} epochs (no improvement for {self.config.patience} epochs)")
                break
        
        # Save final model and history
        self.save('final')
        self._save_history()
        print("\n✓ Training complete!")
    
    def save(self, tag: str = 'checkpoint'):
        """Save model checkpoint"""
        ckpt_path = Path(self.config.save_dir) / f'smrn_{tag}.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, ckpt_path)
    
    def load(self, path: str):
        """Load model checkpoint"""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.history = ckpt.get('history', self.history)
        print(f"Loaded checkpoint from {path}")
    
    def _save_history(self):
        """Save training history as JSON"""
        history_path = Path(self.config.save_dir) / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Also save config
        config_path = Path(self.config.save_dir) / 'config.json'
        config_dict = {k: v for k, v in vars(self.config).items()}
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def main():
    """CLI for training SMRN"""
    parser = argparse.ArgumentParser(description='Train SMRN')
    
    # Model config
    parser.add_argument('--task', type=str, default='recall', 
                       choices=['recall', 'lm', 'haystack', 'listops'])
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--no_rff', action='store_true', help='Disable Random Fourier Features')
    
    # Training config
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    # Data config
    parser.add_argument('--n_samples', type=int, default=20000)
    parser.add_argument('--text_file', type=str, help='Text file for character LM')
    
    # Other
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load data
    print(f"Loading {args.task} dataset...")
    if args.task == 'recall':
        train_loader, val_loader, vocab_size = get_recall_loaders(
            n_samples=args.n_samples,
            seq_len=args.seq_len,
            batch_size=args.batch_size
        )
    elif args.task == 'lm':
        if args.text_file is None:
            raise ValueError("--text_file required for LM task")
        with open(args.text_file, 'r') as f:
            text = f.read()
        train_loader, val_loader, vocab_size, char2idx, idx2char = get_char_loaders(
            text=text,
            seq_len=args.seq_len,
            batch_size=args.batch_size
        )
    elif args.task == 'haystack':
        train_loader, vocab_size = get_haystack_loaders(
            n_samples=args.n_samples,
            batch_size=args.batch_size
        )
        val_loader = train_loader  # Use same for validation
    elif args.task == 'listops':
        dataset = ListOpsDataset(n_samples=args.n_samples)
        n_train = int(0.9 * len(dataset))
        train_data, val_data = torch.utils.data.random_split(
            dataset, [n_train, len(dataset) - n_train]
        )
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        vocab_size = dataset.vocab_size
    
    # Create config
    config = SMRNConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        window_size=args.window_size,
        use_rff=not args.no_rff,
        task=args.task,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        n_samples=args.n_samples,
        save_dir=args.save_dir,
        seed=args.seed
    )
    
    # Create model
    model = SMRN(config)
    
    # Create trainer
    trainer = SMRNTrainer(model, config, device=args.device)
    
    # Resume if specified
    if args.resume:
        trainer.load(args.resume)
    
    # Train
    trainer.fit(train_loader, val_loader, task=args.task)


if __name__ == '__main__':
    main()
