import torch
import torch.nn as nn
import numpy as np
import math

from scripts._bootstrap import ROOT_DIR
from llm.tokenizer import BPETokenizer
from llm.config import GPTConfig, TrainConfig
from llm.mamba import MambaLanguageModel
from llm.paths import (
    MODEL_PATH,
    TOKENIZER_PREFIX,
    TRAIN_BIN_PATH,
    VAL_BIN_PATH,
    ensure_project_dirs,
)

# Configuration
train_config = TrainConfig()
gpt_config = GPTConfig()
# Override for mamba if needed
gpt_config.n_embd = 256
gpt_config.n_layer = 4

torch.manual_seed(1337)

# Load tokenizer
tokenizer = BPETokenizer()
tokenizer.load(str(TOKENIZER_PREFIX))
gpt_config.vocab_size = len(tokenizer.vocab)

def get_lr(it, config):
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
            x, y = x.to(config.device), y.to(config.device)
            
            logits, loss = model(x, y)
            losses[k] = loss.item()
        
        avg_loss = losses.mean()
        out[split] = avg_loss.item()
        out[f'{split}_perplexity'] = math.exp(avg_loss.item())
    model.train()
    return out

def train():
    ensure_project_dirs()
    
    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PREFIX))
    gpt_config.vocab_size = len(tokenizer.vocab)
    
    train_data = np.memmap(TRAIN_BIN_PATH, dtype=np.uint16, mode='r')
    val_data = np.memmap(VAL_BIN_PATH, dtype=np.uint16, mode='r')
    
    model = MambaLanguageModel(gpt_config)
    model.to(train_config.device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=train_config.use_amp)
    
    print(f"Training Mamba on {train_config.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    best_val_loss = float('inf')
    
    for iter in range(train_config.max_iters):
        lr = get_lr(iter, train_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if iter % train_config.eval_interval == 0 or iter == train_config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, train_config)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.6f}")
        
        ix = torch.randint(len(train_data) - gpt_config.block_size, (train_config.batch_size,))
        x = torch.stack([torch.from_numpy((train_data[i:i+gpt_config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((train_data[i+1:i+1+gpt_config.block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(train_config.device), y.to(train_config.device)
        
        with torch.amp.autocast('cuda', enabled=train_config.use_amp):
            logits, loss = model(x, y)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        if train_config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
    
    torch.save(model.state_dict(), MODEL_PATH.parent / "mamba_model.pt")
    print("Training complete!")

if __name__ == "__main__":
    train()
