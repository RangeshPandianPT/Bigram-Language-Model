from dataclasses import dataclass
import torch

@dataclass
class GPTConfig:
    # ── Scaled-up model: ~5M parameters ──────────────────────────────────
    block_size: int = 128          # context length (was 64)
    vocab_size: int = 512          # will be overwritten by tokenizer
    n_layer: int = 8               # depth (was 4)
    n_head: int = 8               # query heads (was 4)
    n_embd: int = 256              # embedding dim (was 128)
    dropout: float = 0.1          # lighter dropout for larger model
    bias: bool = False
    n_kv_head: int = 4            # GQA: 4 KV heads for 8 query heads (50% KV saving)

    def __post_init__(self):
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.n_head % self.n_kv_head == 0, \
            f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"

@dataclass
class TrainConfig:
    batch_size: int = 32           # smaller batch to fit larger model in memory
    max_iters: int = 5000         # more iterations for larger model (was 2000)
    eval_interval: int = 500
    learning_rate: float = 3e-4
    eval_iters: int = 200
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Training improvements
    use_amp: bool = True          # Mixed precision training
    grad_clip: float = 1.0       # Gradient clipping
    weight_decay: float = 0.1    # Weight decay
    warmup_iters: int = 200      # longer warmup for larger model (was 100)
    lr_decay_iters: int = 5000   # aligned with max_iters (was 2000)
    min_lr: float = 3e-5         # Minimum learning rate

@dataclass
class SamplingConfig:
    temperature: float = 1.0  # Higher = more random, lower = more deterministic
    top_k: int = 0  # 0 = disabled, else sample from top k tokens
    top_p: float = 1.0  # 1.0 = disabled, else nucleus sampling
    repetition_penalty: float = 1.0  # 1.0 = no penalty, >1.0 = penalize repetition
