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
