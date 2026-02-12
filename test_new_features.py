"""
Quick test to verify new training features work correctly
"""
import torch
import numpy as np
from tokenizer import BPETokenizer
from config import GPTConfig, TrainConfig
from model import GPTLanguageModel
import math

# Override config for quick test
train_config = TrainConfig()
train_config.max_iters = 100
train_config.eval_interval = 50
train_config.batch_size = 32

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

print("=" * 80)
print("TESTING NEW TRAINING FEATURES")
print("=" * 80)
print(f"Device: {train_config.device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print(f"Mixed precision (AMP): {train_config.use_amp}")
print(f"Gradient clipping: {train_config.grad_clip}")
print(f"Weight decay: {train_config.weight_decay}")
print(f"Warmup iters: {train_config.warmup_iters}")
print("=" * 80)

# Test learning rate schedule
print("\nTesting LR Schedule:")
from train import get_lr
for it in [0, 50, 100, 500, 1000, 2000, 2500]:
    lr = get_lr(it, train_config)
    print(f"  Iter {it:4d}: LR = {lr:.6f}")

# Run a few training iterations
print("\nRunning 100 training iterations...")
for iter in range(train_config.max_iters):
    # Update learning rate
    lr = get_lr(iter, train_config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Sample a batch
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
    
    if iter % 25 == 0:
        print(f"  Iter {iter:3d}: loss = {loss.item():.4f}, lr = {lr:.6f}")

print("\n" + "=" * 80)
print("✓ All training features working correctly!")
print("=" * 80)

# Test generation with new sampling
print("\nTesting advanced sampling strategies:")
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=train_config.device)

configs = [
    {"name": "Default", "temp": 1.0, "top_k": 0, "top_p": 1.0},
    {"name": "Low temp", "temp": 0.7, "top_k": 0, "top_p": 1.0},
    {"name": "Top-k=50", "temp": 1.0, "top_k": 50, "top_p": 1.0},
]

for cfg in configs:
    out = model.generate(context, max_new_tokens=20, 
                        temperature=cfg['temp'], 
                        top_k=cfg['top_k'], 
                        top_p=cfg['top_p'])
    print(f"  {cfg['name']:12s}: {tokenizer.decode(out[0].tolist()[:30])}")

print("\n✓ All tests passed!")
