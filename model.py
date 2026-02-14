import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from config import GPTConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
    # xq.shape = [batch_size, seq_len, n_head, head_dim]
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)
    
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)
    
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    if n_rep == 1:
        return x
    B, T, n_kv_head, head_dim = x.shape
    x = x[:, :, :, None, :].expand(B, T, n_kv_head, n_rep, head_dim)
    return x.reshape(B, T, n_kv_head * n_rep, head_dim)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        
        # Checking if GQA is valid
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=config.bias)
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        
        self.dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cos, freqs_sin, past_kv=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        # calculate query, key, values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(B, T, self.n_head, self.head_dim) # (B, T, nh, hs)
        k = k.view(B, T, self.n_kv_head, self.head_dim) # (B, T, n_kv, hs)
        v = v.view(B, T, self.n_kv_head, self.head_dim) # (B, T, n_kv, hs)

        # Apply RoPE (only to q and k)
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        
        # KV Cache: Concatenate with past keys/values
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=1)  # Concatenate on sequence dimension
            v = torch.cat((past_v, v), dim=1)
            
        present_kv = (k, v) # cache the keys/values for next step
        
        # GQA: Repeat K/V heads to match Q heads
        # This is where GQA happens. If n_kv_head < n_head, we repeat the efficient K/V heads
        k = repeat_kv(k, self.n_head // self.n_kv_head)
        v = repeat_kv(v, self.n_head // self.n_kv_head)
        
        # Transpose for attention: (B, nh, T, hs)
        k = k.transpose(1, 2) 
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Masking - need to handle both cached and non-cached cases
        if past_kv is None:
            # Standard causal masking for training
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        else:
            # During generation with cache, we only compute attention for the new token
            # The new query can attend to all previous keys (no masking needed since we're generating left-to-right)
            pass
            
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, present_kv

class FeedForward(nn.Module):
    """ SwiGLU FeedForward """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # To maintain parameter count roughly similar or follow Llama, we usually do a bit different sizing.
        # But let's stick to 4*n_embd hidden dim for simplicity and power.
        hidden_dim = 4 * config.n_embd
        hidden_dim = int(2 * hidden_dim / 3) 
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.sa = CausalSelfAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = RMSNorm(config.n_embd)
        self.ln2 = RMSNorm(config.n_embd)

    def forward(self, x, freqs_cos, freqs_sin, past_kv=None):
        sa_out, present_kv = self.sa(self.ln1(x), freqs_cos, freqs_sin, past_kv)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, present_kv

class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_embd) # final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Precompute RoPE frequencies
        head_dim = config.n_embd // config.n_head
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, config.block_size)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, idx, targets=None, past_key_values=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        x = tok_emb # (B,T,C)
        
        # Calculate RoPE offset
        if past_key_values is not None:
             past_length = past_key_values[0][0].size(1)  # past_k shape: (B, past_len, n_head, head_dim)
             start_pos = past_length
        else:
             start_pos = 0
             
        freqs_cos = self.freqs_cos[start_pos : start_pos + T]
        freqs_sin = self.freqs_sin[start_pos : start_pos + T]
        
        present_key_values = []
        for i, block in enumerate(self.blocks):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, freqs_cos, freqs_sin, past_kv)
            present_key_values.append(present_kv)
            
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, present_key_values

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0):
        """
        Generate text with advanced sampling strategies.
        
        Args:
            idx: (B, T) tensor of token indices
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: if > 0, only sample from top k tokens
            top_p: if < 1.0, nucleus sampling (sample from smallest set with cumulative prob >= p)
            repetition_penalty: if > 1.0, penalize repeated tokens
        """
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # If we have past_kv, we only need to pass the last token
            if past_key_values is not None:
                idx_cond = idx[:, -1:]
            else:
                # crop idx to the last block_size tokens (legacy behavior if no cache)
                idx_cond = idx[:, -self.config.block_size:] 
                
            # get the predictions
            logits, _, past_key_values = self(idx_cond, past_key_values=past_key_values)
            
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
