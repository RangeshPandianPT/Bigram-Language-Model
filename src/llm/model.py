import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.checkpoint import checkpoint
from .config import GPTConfig

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

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, rope_type: str = "default", rope_scale: float = 1.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    if rope_type == "linear" and rope_scale != 1.0:
        t = torch.arange(end, device=freqs.device).float() / rope_scale
    elif rope_type == "yarn" and rope_scale != 1.0:
        t = torch.arange(end, device=freqs.device).float()
        low_freq_factor = 1.0 / rope_scale
        dims = torch.arange(0, dim, 2)[: (dim // 2)].float()
        smooth = torch.clamp((dims - dim * 0.2) / (dim * 0.4), 0.0, 1.0)
        freq_factor = (1.0 - smooth) * low_freq_factor + smooth * 1.0
        freqs = freqs * freq_factor
    else:
        t = torch.arange(end, device=freqs.device).float()

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
        self.sliding_window = getattr(config, 'sliding_window', None)
        self.use_mla = getattr(config, 'use_mla', False)
        
        if self.use_mla:
            self.kv_lora_rank = getattr(config, 'kv_lora_rank', 64)
            self.q_lora_rank = getattr(config, 'q_lora_rank', 128)
            self.rope_head_dim = getattr(config, 'rope_head_dim', 16)
            self.nope_head_dim = self.head_dim - self.rope_head_dim
            assert self.nope_head_dim > 0, "head_dim must be greater than rope_head_dim for MLA"
            
            self.w_kv_down = nn.Linear(config.n_embd, self.kv_lora_rank, bias=config.bias)
            self.w_kv_up_k = nn.Linear(self.kv_lora_rank, config.n_head * self.nope_head_dim, bias=config.bias)
            self.w_kv_up_v = nn.Linear(self.kv_lora_rank, config.n_head * self.head_dim, bias=config.bias)
            self.w_k_rope = nn.Linear(config.n_embd, config.n_head * self.rope_head_dim, bias=config.bias)
            
            self.w_q_down = nn.Linear(config.n_embd, self.q_lora_rank, bias=config.bias)
            self.w_q_up_nope = nn.Linear(self.q_lora_rank, config.n_head * self.nope_head_dim, bias=config.bias)
            self.w_q_up_rope = nn.Linear(self.q_lora_rank, config.n_head * self.rope_head_dim, bias=config.bias)
        else:
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
        B, T, C = x.size()
        
        if self.use_mla:
            c_kv = self.w_kv_down(x)
            k_nope = self.w_kv_up_k(c_kv).view(B, T, self.n_head, self.nope_head_dim)
            v = self.w_kv_up_v(c_kv).view(B, T, self.n_head, self.head_dim)
            k_rope = self.w_k_rope(x).view(B, T, self.n_head, self.rope_head_dim)
            
            c_q = self.w_q_down(x)
            q_nope = self.w_q_up_nope(c_q).view(B, T, self.n_head, self.nope_head_dim)
            q_rope = self.w_q_up_rope(c_q).view(B, T, self.n_head, self.rope_head_dim)
            
            q_rope, k_rope = apply_rotary_emb(q_rope, k_rope, freqs_cos, freqs_sin)
            
            q = torch.cat([q_nope, q_rope], dim=-1)
            k = torch.cat([k_nope, k_rope], dim=-1)
            
            if past_kv is not None:
                past_k, past_v = past_kv
                k = torch.cat((past_k, k), dim=1)
                v = torch.cat((past_v, v), dim=1)
                
            present_kv = (k, v)
            
            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            
            q = q.view(B, T, self.n_head, self.head_dim)
            k = k.view(B, T, self.n_kv_head, self.head_dim)
            v = v.view(B, T, self.n_kv_head, self.head_dim)

            q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
            
            if past_kv is not None:
                past_k, past_v = past_kv
                k = torch.cat((past_k, k), dim=1)
                v = torch.cat((past_v, v), dim=1)
                
            present_kv = (k, v)
            
            k = repeat_kv(k, self.n_head // self.n_kv_head)
            v = repeat_kv(v, self.n_head // self.n_kv_head)
            
            k = k.transpose(1, 2) 
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)

        if past_kv is None:
            if self.sliding_window is not None:
                mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
                mask = torch.triu(mask, diagonal=-self.sliding_window + 1)
                y = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=mask, 
                    dropout_p=self.dropout.p if self.training else 0.0, 
                    is_causal=False
                )
            else:
                y = F.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=None, 
                    dropout_p=self.dropout.p if self.training else 0.0, 
                    is_causal=True
                )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout.p if self.training else 0.0, 
                is_causal=False
            )
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
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

class MoEBlock(nn.Module):
    """ Mixture of Experts (MoE) block with optional DeepSeek-style Shared Experts """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_shared_experts = getattr(config, 'n_shared_experts', 0)
        
        # Gate network to compute routing probabilities
        self.gate = nn.Linear(config.n_embd, self.n_experts, bias=False)
        
        # Expert networks
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(self.n_experts)])
        
        # Dedicated shared experts processed for all tokens
        if self.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([FeedForward(config) for _ in range(self.n_shared_experts)])
        
    def forward(self, x):
        # x shape: (B, T, C)
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1]) # (B*T, C)
        
        # Shared experts output (processed for every token)
        shared_output = 0.0
        if self.n_shared_experts > 0:
            for expert in self.shared_experts:
                shared_output = shared_output + expert(x)
        
        # Get routing logits
        router_logits = self.gate(x_flat) # (B*T, n_experts)
        
        # Softmax to get routing probabilities
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        
        # Get top-k experts and their probabilities
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        
        # Normalize routing weights so they sum to 1 over the selected top-k
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)
        
        final_hidden_states = torch.zeros_like(x_flat)
        
        # One-hot encode the selected experts to create masks: (B*T, top_k, n_experts)
        expert_mask = F.one_hot(selected_experts, num_classes=self.n_experts).permute(2, 1, 0)
        
        for expert_idx in range(self.n_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            if top_x.shape[0] == 0:
                continue
                
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            
            current_state = x_flat[top_x_list]
            current_hidden_states = expert_layer(current_state)
            current_hidden_states = current_hidden_states * routing_weights[top_x_list, idx_list, None]
            
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))
            
        routed_out = final_hidden_states.view(orig_shape)
        if self.n_shared_experts > 0:
            return routed_out + shared_output
        return routed_out

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.sa = CausalSelfAttention(config)
        if getattr(config, 'n_experts', 0) > 0:
            self.ffwd = MoEBlock(config)
        else:
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
        
        # Weight tying
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding_table.weight
        
        # Precompute RoPE frequencies
        head_dim = (
            config.rope_head_dim 
            if getattr(config, 'use_mla', False) 
            else config.n_embd // config.n_head
        )
        rope_type = getattr(config, 'rope_type', 'default')
        rope_scale = getattr(config, 'rope_scale', 1.0)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            head_dim, config.block_size, rope_type=rope_type, rope_scale=rope_scale
        )
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
            
            if self.training and getattr(self.config, 'gradient_checkpointing', False) and past_kv is None:
                x, present_kv = checkpoint(block, x, freqs_cos, freqs_sin, past_kv, use_reentrant=False)
            else:
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
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0):
        """
        Generate text with advanced sampling strategies.
        
        Args:
            idx: (B, T) tensor of token indices
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: if > 0, only sample from top k tokens
            top_p: if < 1.0, nucleus sampling (sample from smallest set with cumulative prob >= p)
            min_p: if > 0.0, sample from tokens with prob >= min_p * max(prob)
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
            
            # Apply min-p filtering
            if min_p > 0.0:
                probs = F.softmax(logits, dim=-1)
                max_probs = probs.max(dim=-1, keepdim=True).values
                tokens_to_remove = probs < (min_p * max_probs)
                logits[tokens_to_remove] = -float('Inf')
            
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

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0, repetition_penalty=1.0):
        past_key_values = None
        
        for _ in range(max_new_tokens):
            if past_key_values is not None:
                idx_cond = idx[:, -1:]
            else:
                idx_cond = idx[:, -self.config.block_size:] 
                
            logits, _, past_key_values = self(idx_cond, past_key_values=past_key_values)
            logits = logits[:, -1, :] 
            
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    logits[0, token_id] /= repetition_penalty
            
            logits = logits / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            if min_p > 0.0:
                probs = F.softmax(logits, dim=-1)
                max_probs = probs.max(dim=-1, keepdim=True).values
                tokens_to_remove = probs < (min_p * max_probs)
                logits[tokens_to_remove] = -float('Inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next
