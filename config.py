from dataclasses import dataclass
import torch

@dataclass
class GPTConfig:
    block_size: int = 64
    vocab_size: int = 512 # will be overwritten by tokenizer
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.2
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better like Llama.

@dataclass
class TrainConfig:
    batch_size: int = 64
    max_iters: int = 2000
    eval_interval: int = 200
    learning_rate: float = 3e-4
    eval_iters: int = 200
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Training improvements
    use_amp: bool = True  # Mixed precision training
    grad_clip: float = 1.0  # Gradient clipping
    weight_decay: float = 0.1  # Weight decay
    warmup_iters: int = 100  # LR warmup steps
    lr_decay_iters: int = 2000  # LR decay steps
    min_lr: float = 3e-5  # Minimum learning rate

@dataclass
class SamplingConfig:
    temperature: float = 1.0  # Higher = more random, lower = more deterministic
    top_k: int = 0  # 0 = disabled, else sample from top k tokens
    top_p: float = 1.0  # 1.0 = disabled, else nucleus sampling
    repetition_penalty: float = 1.0  # 1.0 = no penalty, >1.0 = penalize repetition
