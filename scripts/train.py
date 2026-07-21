import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import numpy as np
import os
import math

from scripts._bootstrap import ROOT_DIR
from llm.tokenizer import BPETokenizer
from llm.config import GPTConfig, TrainConfig
from llm.model import GPTLanguageModel
from llm.experiment import save_json, set_seed, utc_now_iso
from llm.paths import (
    EVAL_RESULTS_PATH,
    MODEL_BEST_PATH,
    MODEL_PATH,
    TRAINING_METADATA_PATH,
    TOKENIZER_PREFIX,
    TRAIN_BIN_PATH,
    VAL_BIN_PATH,
    ensure_project_dirs,
)

# Configuration
train_config = TrainConfig()
gpt_config = GPTConfig()

set_seed(train_config.seed)

# Load tokenizer
tokenizer = BPETokenizer()
tokenizer.load(str(TOKENIZER_PREFIX))
gpt_config.vocab_size = len(tokenizer.vocab)

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

def train(model=None, train_config=None, gpt_config=None):
    ensure_project_dirs()

    # Load config
    if train_config is None:
        train_config = TrainConfig()
    if gpt_config is None:
        gpt_config = getattr(model, 'config', GPTConfig()) if model is not None else GPTConfig()

    if hasattr(train_config, "seed"):
        set_seed(train_config.seed)
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PREFIX))
    if gpt_config is not None:
        gpt_config.vocab_size = len(tokenizer.vocab)
    
    # Load data
    train_data = np.memmap(TRAIN_BIN_PATH, dtype=np.uint16, mode='r')
    val_data = np.memmap(VAL_BIN_PATH, dtype=np.uint16, mode='r')
    
    # Setup DDP
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend='nccl' if torch.cuda.is_available() and os.name != 'nt' else 'gloo')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        master_process = True
        device = train_config.device
        ddp_world_size = 1
        ddp_local_rank = 0

    # Initialize model
    if model is None:
        model = GPTLanguageModel(gpt_config)
    
    model.to(device)
    
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    elif torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
        
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )
    
    # Mixed precision training
    scaler = torch.amp.GradScaler('cuda', enabled=train_config.use_amp)
    
    if master_process:
        print(f"Training on {device} (DDP: {ddp}, World Size: {ddp_world_size})")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print(f"Mixed precision: {train_config.use_amp}")

        save_json(
            TRAINING_METADATA_PATH,
            {
                "created_at": utc_now_iso(),
                "seed": train_config.seed,
                "device": device,
                "ddp": ddp,
                "world_size": ddp_world_size,
                "train_config": train_config,
                "gpt_config": gpt_config,
                "tokenizer_vocab_size": len(tokenizer.vocab),
                "train_data_path": str(TRAIN_BIN_PATH),
                "val_data_path": str(VAL_BIN_PATH),
                "best_checkpoint": str(MODEL_BEST_PATH),
                "final_checkpoint": str(MODEL_PATH),
            },
        )
    
    best_val_loss = float('inf')
    
    for iter in range(train_config.max_iters):
        # Update learning rate
        lr = get_lr(iter, train_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate periodically
        if iter % train_config.eval_interval == 0 or iter == train_config.max_iters - 1:
            if master_process:
                losses = estimate_loss(model, train_data, val_data, train_config)
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                      f"train ppl {losses['train_perplexity']:.2f}, val ppl {losses['val_perplexity']:.2f}, lr {lr:.6f}")
                
                # Save best model
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), MODEL_BEST_PATH)
                    save_json(
                        MODEL_BEST_PATH.with_suffix('.meta.json'),
                        {
                            "saved_at": utc_now_iso(),
                            "iteration": iter,
                            "split_metrics": losses,
                            "learning_rate": lr,
                            "seed": train_config.seed,
                            "checkpoint": str(MODEL_BEST_PATH),
                        },
                    )
                    print(f"Saved best model with val loss {best_val_loss:.4f}")
        
        # Sample a batch of data
        block_size = model.module.config.block_size if hasattr(model, 'module') else model.config.block_size
        ix = torch.randint(len(train_data) - block_size, (train_config.batch_size,))
        x = torch.stack([torch.from_numpy((train_data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((train_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda', enabled=train_config.use_amp):
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
    
    if master_process:
        # Save final model
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), MODEL_PATH)
        save_json(
            MODEL_PATH.with_suffix('.meta.json'),
            {
                "saved_at": utc_now_iso(),
                "final_iteration": train_config.max_iters - 1,
                "best_val_loss": best_val_loss,
                "seed": train_config.seed,
                "checkpoint": str(MODEL_PATH),
                "training_metadata": str(TRAINING_METADATA_PATH),
                "eval_results": str(EVAL_RESULTS_PATH),
            },
        )
        print("Training complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    train()

