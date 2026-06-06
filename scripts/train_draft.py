import sys
import os
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm.config import GPTConfig, TrainConfig
from llm.model import GPTLanguageModel
from llm.paths import MODEL_PATH

def train_draft_model():
    print("Training tiny draft model for speculative decoding...")
    
    # 1. Config for a very small model
    draft_config = GPTConfig(n_layer=2, n_embd=64, n_head=2, block_size=64)
    # Get vocab size from the main config/tokenizer context (usually 512 for dummy BPE, let's assume 512 for safety)
    draft_config.vocab_size = 512 
    
    train_config = TrainConfig(batch_size=8, max_iters=200, eval_interval=50)
    device = train_config.device
    
    model = GPTLanguageModel(draft_config)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Dummy training loop using random data
    # In reality, this would be trained on the same data distribution as the main model
    model.train()
    for iter in range(train_config.max_iters):
        xb = torch.randint(0, draft_config.vocab_size, (train_config.batch_size, draft_config.block_size)).to(device)
        yb = torch.randint(0, draft_config.vocab_size, (train_config.batch_size, draft_config.block_size)).to(device)
        
        logits, loss, _ = model(xb, targets=yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % train_config.eval_interval == 0:
            print(f"Iter {iter}: loss {loss.item():.4f}")
            
    # Save the draft model
    draft_path = MODEL_PATH.parent / "draft_model.pt"
    os.makedirs(draft_path.parent, exist_ok=True)
    torch.save(model.state_dict(), draft_path)
    print(f"Draft model saved to {draft_path}")

if __name__ == "__main__":
    train_draft_model()
