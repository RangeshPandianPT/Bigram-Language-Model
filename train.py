import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizer import BPETokenizer
from config import GPTConfig, TrainConfig
from model import GPTLanguageModel

# Configuration
train_config = TrainConfig()
gpt_config = GPTConfig()

torch.manual_seed(1337)

# Load tokenizer
tokenizer = BPETokenizer()
tokenizer.load("bpe")
gpt_config.vocab_size = len(tokenizer.vocab)

import numpy as np
import os
import math

def get_lr(it, config):
    """Learning rate schedule with warmup and cosine decay"""
    # 1) Linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) If it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    """Estimate loss and perplexity on train and val sets"""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
            x, y = x.to(config.device), y.to(config.device)
            
            logits, loss, _ = model(x, y)
            losses[k] = loss.item()
        
        avg_loss = losses.mean()
        out[split] = avg_loss.item()
        out[f'{split}_perplexity'] = math.exp(avg_loss.item())
    model.train()
    return out

def train():
    # Load config
    train_config = TrainConfig()
    gpt_config = GPTConfig()
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load("bpe")
    gpt_config.vocab_size = len(tokenizer.vocab)
    
    # Load data
    train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')
    
    # Initialize model
    model = GPTLanguageModel(gpt_config)
    model.to(train_config.device)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=train_config.use_amp)
    
    print(f"Training on {train_config.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Mixed precision: {train_config.use_amp}")
    
    best_val_loss = float('inf')
    
    for iter in range(train_config.max_iters):
        # Update learning rate
        lr = get_lr(iter, train_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate periodically
        if iter % train_config.eval_interval == 0 or iter == train_config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, train_config)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                  f"train ppl {losses['train_perplexity']:.2f}, val ppl {losses['val_perplexity']:.2f}, lr {lr:.6f}")
            
            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), 'model_best.pth')
                print(f"Saved best model with val loss {best_val_loss:.4f}")
        
        # Sample a batch of data
        ix = torch.randint(len(train_data) - gpt_config.block_size, (train_config.batch_size,))
        x = torch.stack([torch.from_numpy((train_data[i:i+gpt_config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((train_data[i+1:i+1+gpt_config.block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(train_config.device), y.to(train_config.device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=train_config.use_amp):
            logits, loss, _ = model(x, y)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if train_config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
    
    # Save final model
    torch.save(model.state_dict(), 'model.pth')
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()

