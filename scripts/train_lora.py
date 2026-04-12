import torch
import numpy as np
import os

from scripts._bootstrap import ROOT_DIR
from scripts.train import get_lr
from llm.tokenizer import BPETokenizer
from llm.config import GPTConfig, TrainConfig
from llm.model import GPTLanguageModel
from llm.lora import inject_lora
from llm.paths import (
    LORA_WEIGHTS_PATH,
    MODEL_PATH,
    TOKENIZER_PREFIX,
    TRAIN_BIN_PATH,
    VAL_BIN_PATH,
    ensure_project_dirs,
)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, train_config, block_size):
    """Estimate loss on train and val sets"""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(train_config.eval_iters)
        for k in range(train_config.eval_iters):
            ix = torch.randint(len(data) - block_size, (train_config.batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
            x, y = x.to(train_config.device), y.to(train_config.device)
            
            with torch.cuda.amp.autocast(enabled=train_config.use_amp):
                logits, loss, _ = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def train_lora():
    ensure_project_dirs()

    train_config = TrainConfig()
    
    # We will run a quick training loop for demonstration
    train_config.max_iters = 1000
    train_config.eval_interval = 200
    train_config.learning_rate = 1e-3 # LoRA needs higher LR
    
    print("Loading config and tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PREFIX))
    
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}' to fine-tune.")
        return
        
    print(f"Loading base weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    config = GPTConfig()
    
    if 'token_embedding_table.weight' in checkpoint:
        config.vocab_size, config.n_embd = checkpoint['token_embedding_table.weight'].shape
    n_layer = sum(1 for k in checkpoint if k.endswith('.sa.c_proj.weight'))
    if n_layer > 0: config.n_layer = n_layer
    if config.n_embd == 128:
        config.n_head = 4; config.n_kv_head = 2; config.block_size = 64
    elif config.n_embd == 256:
        config.n_head = 8; config.n_kv_head = 4; config.block_size = 128
        
    # Enable LoRA
    config.lora_rank = 8
    config.lora_alpha = 32
    config.lora_dropout = 0.05
    
    print(f"Initializing base model with vocab_size={config.vocab_size}, n_embd={config.n_embd}, n_layer={config.n_layer}")
    model = GPTLanguageModel(config)
    
    # strict=False because original state_dict doesn't contain LoRA weights
    model.load_state_dict(checkpoint, strict=False)
    
    # Inject LoRA
    print(f"Injecting LoRA with rank {config.lora_rank}...")
    model = inject_lora(model, config)
    
    # Only train LoRA parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params} || Total params: {total_params} || %: {100 * trainable_params / total_params:.2f}%")
    
    model.to(train_config.device)
    
    # Load data
    train_data = np.memmap(TRAIN_BIN_PATH, dtype=np.uint16, mode='r')
    val_data = np.memmap(VAL_BIN_PATH, dtype=np.uint16, mode='r')
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=train_config.use_amp)
    
    print("Starting LoRA fine-tuning...")
    best_val_loss = float('inf')
    
    for iter in range(train_config.max_iters):
        lr = get_lr(iter, train_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        if iter % train_config.eval_interval == 0 or iter == train_config.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, train_config, config.block_size)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                # Save only LoRA parameters
                lora_state_dict = {n: p for n, p in model.state_dict().items() if 'lora_' in n}
                torch.save(lora_state_dict, LORA_WEIGHTS_PATH)
                
        ix = torch.randint(len(train_data) - config.block_size, (train_config.batch_size,))
        x = torch.stack([torch.from_numpy((train_data[i:i+config.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((train_data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
        x, y = x.to(train_config.device), y.to(train_config.device)
        
        with torch.cuda.amp.autocast(enabled=train_config.use_amp):
            logits, loss, _ = model(x, y)
            
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if train_config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        
    print("LoRA Fine-tuning complete!")

if __name__ == "__main__":
    train_lora()
