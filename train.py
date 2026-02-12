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

# Load data using numpy memmap
train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')

# Data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - gpt_config.block_size, (train_config.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+gpt_config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+gpt_config.block_size+1]).astype(np.int64)) for i in ix])
    return x.to(train_config.device), y.to(train_config.device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(train_config.eval_iters)
        for k in range(train_config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == '__main__':
    model = GPTLanguageModel(gpt_config)
    m = model.to(train_config.device)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)

    # Initialize GradScaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(train_config.device == 'cuda'))

    print(f"Device: {train_config.device}")
    print(f"Model parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    print("Starting training...")

    for iter in range(train_config.max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % train_config.eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss with mixed precision
        with torch.amp.autocast(device_type=train_config.device, dtype=torch.float16, enabled=(train_config.device == 'cuda')):
            logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Save the model
    torch.save(m.state_dict(), 'model.pth')
    print("Model saved to model.pth")

    # generate from the model
    print("\nGenerating text:")
    context = torch.zeros((1, 1), dtype=torch.long, device=train_config.device)
    print(tokenizer.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
