"""Selective Memory and Recall Network (SMRN) - PyTorch Implementation

Paper: "Selective Memory and Recall Network: A Theoretical Analysis for Linear-Time Agents"
Authors: Chirag Goyal, Manoj Kumar — VIT Bhopal University

This implementation includes:
- Component 1: SelectiveSSM (Pathway A - Compression) with Mamba-style selectivity
- Component 2: LinearAttentionPathway (Pathway B - Recall) with O(N) complexity
- Component 3: EntropyGate (Dynamic Gating) based on contextual entropy
- Full SMRN model with ablation variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from einops import rearrange, repeat


@dataclass
class SMRNConfig:
    """Configuration for SMRN model"""
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    d_state: int = 16  # SSM state dimension
    window_size: int = 8  # For entropy calculation
    dropout: float = 0.1
    use_rff: bool = True  # Use Random Fourier Features for linear attention
    
    # Task-specific
    task: str = 'lm'  # 'recall', 'lm', 'haystack', 'listops'
    batch_size: int = 64
    seq_len: int = 256
    
    # Training
    max_epochs: int = 50
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # For Theorem 4
    warmup_steps: int = 500
    use_amp: bool = True
    
    # Data
    n_samples: int = 20000
    save_dir: str = 'checkpoints'
    log_every: int = 50
    patience: int = 10
    seed: int = 42


class SelectiveSSM(nn.Module):
    """Component 1: Selective State Space Model (Pathway A - Compression)
    
    Continuous-time SSM:
        dh/dt = A(t)h(t) + B(t)x(t)  [State equation]
        y(t) = C(t)h(t) + D(t)x(t)   [Output equation]
    
    Zero-Order Hold (ZOH) discretization:
        h_t = Ã_t * h_{t-1} + B̃_t * x_t
        where Ã_t = exp(Δ*A), B̃_t = Δ*B_t
    
    Selective (input-dependent): Ã_t and B̃_t are functions of x_t (Mamba-style)
    A matrix is log-parameterized for stability
    
    Theorem 1: O(N) time complexity per sequence
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)  # As specified
        
        # A matrix: log-parameterized for stability (ensures ||Ã|| ≤ 1 for Theorem 4)
        # Initialize as negative to ensure exp(Δ*A) < 1
        A = torch.randn(d_model, d_state)
        self.A_log = nn.Parameter(torch.log(torch.abs(A) + 1e-4))  # log-space
        
        # Input-dependent projections (selective mechanism)
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        
        # Output projection (C matrix)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with selective SSM
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # A matrix from log-space (ensures stability)
        A = -torch.exp(self.A_log)  # (d_model, d_state), negative for stability
        
        # Selective mechanism: compute Δ, B from input
        x_proj = self.x_proj(x)  # (batch, seq_len, dt_rank + d_state)
        delta, B = torch.split(x_proj, [self.dt_rank, self.d_state], dim=-1)
        
        # Δ ∈ (0, ∞) via softplus, then project to d_model
        delta = F.softplus(self.dt_proj(delta))  # (batch, seq_len, d_model)
        
        # C projection
        C = self.C_proj(x)  # (batch, seq_len, d_state)
        
        # Discretization: Ã_t = exp(Δ*A), B̃_t = Δ*B
        # A: (d_model, d_state), delta: (batch, seq_len, d_model)
        # Ã_t computation for each position
        outputs = []
        h = torch.zeros(batch, d_model, self.d_state, device=x.device)  # Initial state
        
        for t in range(seq_len):
            # Get current timestep values
            delta_t = delta[:, t, :]  # (batch, d_model)
            B_t = B[:, t, :]  # (batch, d_state)
            C_t = C[:, t, :]  # (batch, d_state)
            x_t = x[:, t, :]  # (batch, d_model)
            
            # Discretization: Ã = exp(Δ * A)
            # delta_t: (batch, d_model, 1) * A: (d_model, d_state) -> (batch, d_model, d_state)
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))  # (batch, d_model, d_state)
            
            # B̃ = Δ * B
            B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (batch, d_model, d_state)
            
            # State update: h_t = Ã_t * h_{t-1} + B̃_t * x_t
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            
            # Output: y_t = C_t * h_t
            y_t = torch.einsum('bn,bmn->bm', C_t, h)  # (batch, d_model)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        return y


class RandomFourierFeatures(nn.Module):
    """Random Fourier Features for linear attention (Theorem 2)
    
    φ(x) = sqrt(2/m) * [cos(ωx), sin(ωx)]
    where ω ~ N(0, σ²I) are random frequencies
    
    This provides orthogonal feature maps for exact associative recall.
    """
    def __init__(self, d_model: int, n_features: int = None):
        super().__init__()
        if n_features is None:
            n_features = d_model
        self.n_features = n_features
        # Random frequencies (fixed, not trainable)
        self.register_buffer('omega', torch.randn(d_model, n_features) * 0.5)
        self.scale = math.sqrt(2.0 / n_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Random Fourier Features
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            φ(x): (batch, seq_len, 2*n_features)
        """
        proj = x @ self.omega  # (batch, seq_len, n_features)
        return self.scale * torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class LinearAttentionPathway(nn.Module):
    """Component 2: Linear Attention Pathway (Pathway B - Recall)
    
    Standard attention O(N²): Attn(Q,K,V) = softmax(QK^T)V
    
    Reformulated as O(N) via kernel trick:
        φ(Q)(φ(K)^T V) / φ(Q)(φ(K)^T 1)
    
    Recurrent state form (Theorem 2 - Unbounded Associative Recall):
        S_t = S_{t-1} + φ(K_t)^T V_t     [Associative memory]
        Z_t = Z_{t-1} + φ(K_t)           [Normalizer]
        γ_t = φ(Q_t) S_t / (φ(Q_t) Z_t)  [Recall]
    
    Under orthogonal φ (RFF), can store O(dk*dv) associations exactly.
    """
    def __init__(self, d_model: int, use_rff: bool = True):
        super().__init__()
        self.d_model = d_model
        self.dk = d_model
        self.dv = d_model
        self.use_rff = use_rff
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, self.dk, bias=False)
        self.k_proj = nn.Linear(d_model, self.dk, bias=False)
        self.v_proj = nn.Linear(d_model, self.dv, bias=False)
        self.out_proj = nn.Linear(self.dv, d_model, bias=False)
        
        # Feature map φ
        if use_rff:
            self.feature_map = RandomFourierFeatures(self.dk)
            self.feature_dim = 2 * self.dk
        else:
            self.feature_map = None
            self.feature_dim = self.dk
    
    def _apply_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature map φ(x)"""
        if self.feature_map is not None:
            return self.feature_map(x)
        else:
            # Simple ELU(x) + 1 (ensures positive for stability)
            return F.elu(x) + 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with linear attention
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch, seq_len, dk)
        K = self.k_proj(x)  # (batch, seq_len, dk)
        V = self.v_proj(x)  # (batch, seq_len, dv)
        
        # Apply feature map
        Q_phi = self._apply_feature_map(Q)  # (batch, seq_len, feature_dim)
        K_phi = self._apply_feature_map(K)  # (batch, seq_len, feature_dim)
        
        # Recurrent form for O(N) complexity
        # S_t = S_{t-1} + φ(K_t)^T V_t
        # Z_t = Z_{t-1} + φ(K_t)
        S = torch.zeros(batch, self.feature_dim, self.dv, device=x.device)
        Z = torch.zeros(batch, self.feature_dim, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            # Update associative memory and normalizer
            K_phi_t = K_phi[:, t, :]  # (batch, feature_dim)
            V_t = V[:, t, :]  # (batch, dv)
            
            S = S + torch.einsum('bf,bd->bfd', K_phi_t, V_t)  # (batch, feature_dim, dv)
            Z = Z + K_phi_t  # (batch, feature_dim)
            
            # Recall: γ_t = φ(Q_t) S_t / φ(Q_t) Z_t
            Q_phi_t = Q_phi[:, t, :]  # (batch, feature_dim)
            numerator = torch.einsum('bf,bfd->bd', Q_phi_t, S)  # (batch, dv)
            denominator = torch.einsum('bf,bf->b', Q_phi_t, Z)  # (batch,)
            
            # Avoid division by zero
            gamma_t = numerator / (denominator.unsqueeze(-1) + 1e-6)
            outputs.append(gamma_t)
        
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, dv)
        y = self.out_proj(y)
        return y


class EntropyGate(nn.Module):
    """Component 3: Entropy-Based Dynamic Gating
    
    Contextual entropy: H_t = -Σ p(x_i) log₂(p(x_i)) over window w
    where p estimated via softmax of token magnitudes
    
    Gate: g_t = σ(W_g [y_ssm; y_attn; H_t] + b_g)
    Output: y_out = g_t ⊙ y_ssm + (1 - g_t) ⊙ y_attn
    
    High entropy (surprising) → trust recall (attention)
    Low entropy (predictable) → trust compression (SSM)
    """
    def __init__(self, d_model: int, window_size: int = 8):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        
        # Gate network: takes [y_ssm, y_attn, entropy] → gate value
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model + 1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
    
    def _compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute contextual entropy H_t = -Σ p(x_i) log₂(p(x_i))
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            H: (batch, seq_len, 1)
        """
        batch, seq_len, d_model = x.shape
        entropies = []
        
        for t in range(seq_len):
            # Get window [max(0, t-w), t]
            start = max(0, t - self.window_size + 1)
            window = x[:, start:t+1, :]  # (batch, window_len, d_model)
            
            # Estimate p via softmax of magnitudes
            magnitudes = torch.norm(window, dim=-1)  # (batch, window_len)
            p = F.softmax(magnitudes, dim=-1)  # (batch, window_len)
            
            # H = -Σ p log₂(p)
            H = -torch.sum(p * torch.log2(p + 1e-9), dim=-1, keepdim=True)  # (batch, 1)
            entropies.append(H)
        
        return torch.stack(entropies, dim=1)  # (batch, seq_len, 1)
    
    def forward(self, y_ssm: torch.Tensor, y_attn: torch.Tensor, x: torch.Tensor, 
                return_gate_values: bool = False) -> torch.Tensor:
        """Apply entropy-based gating
        
        Args:
            y_ssm: Output from SSM pathway (batch, seq_len, d_model)
            y_attn: Output from attention pathway (batch, seq_len, d_model)
            x: Input for entropy computation (batch, seq_len, d_model)
            return_gate_values: If True, return (output, gate_values)
        Returns:
            y_out: Gated output (batch, seq_len, d_model)
        """
        # Compute contextual entropy
        H = self._compute_entropy(x)  # (batch, seq_len, 1)
        
        # Concatenate inputs for gate network
        gate_input = torch.cat([y_ssm, y_attn, H], dim=-1)  # (batch, seq_len, 2*d_model + 1)
        
        # Compute gate values g_t ∈ [0, 1]
        g = self.gate_net(gate_input)  # (batch, seq_len, d_model)
        
        # Gated output: g_t ⊙ y_ssm + (1 - g_t) ⊙ y_attn
        y_out = g * y_ssm + (1 - g) * y_attn
        
        if return_gate_values:
            return y_out, g
        return y_out


class SMRNBlock(nn.Module):
    """Single SMRN block: SSM + LinearAttn + EntropyGate + LayerNorm + FFN"""
    def __init__(self, config: SMRNConfig):
        super().__init__()
        self.config = config
        
        # Component 1: SelectiveSSM
        self.ssm = SelectiveSSM(config.d_model, config.d_state)
        
        # Component 2: LinearAttentionPathway
        self.attn = LinearAttentionPathway(config.d_model, config.use_rff)
        
        # Component 3: EntropyGate
        self.gate = EntropyGate(config.d_model, config.window_size)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x: torch.Tensor, return_gate_values: bool = False):
        """Forward pass through SMRN block"""
        # Store input for residual and entropy computation
        residual = x
        
        # Parallel pathways
        y_ssm = self.ssm(x)
        y_attn = self.attn(x)
        
        # Entropy-based gating
        if return_gate_values:
            y_gated, gate_values = self.gate(y_ssm, y_attn, x, return_gate_values=True)
        else:
            y_gated = self.gate(y_ssm, y_attn, x)
            gate_values = None
        
        # Residual + LayerNorm
        x = self.norm1(residual + y_gated)
        
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        
        if return_gate_values:
            return x, gate_values
        return x


class SMRN(nn.Module):
    """Full Selective Memory and Recall Network
    
    Theorem 3 (Complexity):
        - Time: O(N) per sequence
        - Memory: O(d²) independent of sequence length N
    
    Theorem 4 (Gradient Stability):
        - |∂h_t/∂h_{t-1}| = ||Ã_t|| ≤ 1 (A is negative)
        - Gate outputs ∈ [0,1] (sigmoid bounded)
    """
    def __init__(self, config: SMRNConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, config.seq_len, config.d_model) * 0.02)
        
        # N stacked SMRN blocks
        self.layers = nn.ModuleList([SMRNBlock(config) for _ in range(config.n_layers)])
        
        # Output layers
        self.norm_out = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # LM head (weight-tied with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # Weight tying
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, return_gate_values: bool = False):
        """Forward pass
        
        Args:
            x: (batch, seq_len) token indices
            return_gate_values: If True, return gate activations per layer
        Returns:
            logits: (batch, seq_len, vocab_size)
            gate_values: List of (batch, seq_len, d_model) if return_gate_values=True
        """
        batch, seq_len = x.shape
        
        # Embedding: token + positional
        x = self.token_embed(x)  # (batch, seq_len, d_model)
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)
        
        # Pass through SMRN blocks
        gate_values_list = []
        for layer in self.layers:
            if return_gate_values:
                x, gate_vals = layer(x, return_gate_values=True)
                gate_values_list.append(gate_vals)
            else:
                x = layer(x)
        
        # Output
        x = self.norm_out(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        if return_gate_values:
            return logits, gate_values_list
        return logits
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int = 40, top_p: float = 0.9) -> torch.Tensor:
        """Generate text autoregressively
        
        Args:
            idx: (batch, seq_len) initial context
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus (top-p) filtering
        Returns:
            generated: (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to max seq_len
            idx_cond = idx if idx.size(1) <= self.config.seq_len else idx[:, -self.config.seq_len:]
            
            # Forward pass
            logits = self(idx_cond)  # (batch, seq_len, vocab_size)
            logits = logits[:, -1, :] / temperature  # (batch, vocab_size)
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        
        return idx


class SMRNSSMOnly(nn.Module):
    """Ablation: SSM pathway only (no attention, no gate)"""
    def __init__(self, config: SMRNConfig):
        super().__init__()
        self.config = config
        
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, config.seq_len, config.d_model) * 0.02)
        
        # Only SSM layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'ssm': SelectiveSSM(config.d_model, config.d_state),
                'norm': nn.LayerNorm(config.d_model),
                'ffn': nn.Sequential(
                    nn.Linear(config.d_model, 4 * config.d_model),
                    nn.GELU(),
                    nn.Linear(4 * config.d_model, config.d_model)
                )
            }) for _ in range(config.n_layers)
        ])
        
        self.norm_out = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
    
    def forward(self, x: torch.Tensor):
        x = self.token_embed(x) + self.pos_embed[:, :x.size(1), :]
        for layer in self.layers:
            x = layer['norm'](x + layer['ssm'](x))
            x = x + layer['ffn'](x)
        return self.lm_head(self.norm_out(x))


class SMRNAttnOnly(nn.Module):
    """Ablation: Linear attention only (no SSM, no gate)"""
    def __init__(self, config: SMRNConfig):
        super().__init__()
        self.config = config
        
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, config.seq_len, config.d_model) * 0.02)
        
        # Only attention layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': LinearAttentionPathway(config.d_model, config.use_rff),
                'norm': nn.LayerNorm(config.d_model),
                'ffn': nn.Sequential(
                    nn.Linear(config.d_model, 4 * config.d_model),
                    nn.GELU(),
                    nn.Linear(4 * config.d_model, config.d_model)
                )
            }) for _ in range(config.n_layers)
        ])
        
        self.norm_out = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight
    
    def forward(self, x: torch.Tensor):
        x = self.token_embed(x) + self.pos_embed[:, :x.size(1), :]
        for layer in self.layers:
            x = layer['norm'](x + layer['attn'](x))
            x = x + layer['ffn'](x)
        return self.lm_head(self.norm_out(x))


if __name__ == '__main__':
    # Quick test
    print("Testing SMRN architecture...")
    
    config = SMRNConfig(
        vocab_size=512,
        d_model=256,
        n_layers=4,
        seq_len=128
    )
    
    model = SMRN(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test forward pass
    x = torch.randint(0, config.vocab_size, (2, 64))
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test with gate values
    logits, gates = model(x, return_gate_values=True)
    print(f"Gate values per layer: {len(gates)}")
    print(f"Gate shape: {gates[0].shape}")
    
    print("\n✓ SMRN architecture test passed!")
