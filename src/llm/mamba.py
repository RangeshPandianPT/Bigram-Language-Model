import torch
import torch.nn as nn
from llm.config import GPTConfig

class MambaBlock(nn.Module):
    """
    Simplified Mamba (State Space Model) Block.
    Provides linear time complexity with respect to sequence length.
    """
    def __init__(self, config: GPTConfig, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = config.n_embd
        self.d_inner = int(expand * self.d_model)
        self.d_state = d_state
        
        # In projection
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=config.bias)
        
        # Conv1d for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        # State Space parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False) # delta, B, C
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # A matrix (transition)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Out projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)
        
    def forward(self, hidden_states):
        B, L, D = hidden_states.shape
        
        # Project and split
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        
        # Conv
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        x = torch.nn.functional.silu(x)
        
        # SSM projection
        x_dbl = self.x_proj(x) # (B, L, dt_rank + 2*d_state)
        delta, B_mat, C_mat = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)
        
        # Softplus delta
        delta = torch.nn.functional.softplus(self.dt_proj(delta)) # (B, L, d_inner)
        
        # Simplified SSM computation (O(L) loop in PyTorch, slow but illustrative)
        # In reality, this is done with parallel associative scan in custom CUDA kernels
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        
        y = torch.zeros_like(x)
        states = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        
        for t in range(L):
            xt = x[:, t] # (B, d_inner)
            dt = delta[:, t] # (B, d_inner)
            Bt = B_mat[:, t] # (B, d_state)
            Ct = C_mat[:, t] # (B, d_state)
            
            # Discretize
            dA = torch.exp(dt.unsqueeze(-1) * A) # (B, d_inner, d_state)
            dB = dt.unsqueeze(-1) * Bt.unsqueeze(1) # (B, d_inner, d_state)
            
            # State update
            states = states * dA + dB * xt.unsqueeze(-1) # (B, d_inner, d_state)
            
            # Output
            yt = (states * Ct.unsqueeze(1)).sum(-1) + self.D * xt # (B, d_inner)
            y[:, t] = yt
            
        # Gating
        y = y * torch.nn.functional.silu(z)
        
        # Output projection
        out = self.out_proj(y)
        return out

class MambaLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(config.n_embd),
                MambaBlock(config)
            )
            for _ in range(config.n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding_table.weight
            
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding_table(idx)
        
        for block in self.blocks:
            norm_x, mamba = block[0], block[1]
            x = x + mamba(norm_x(x))
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            return logits, loss
