import torch
import time
import torch.nn.functional as F
from config import GPTConfig
from model import GPTLanguageModel

def test_kv_cache():
    # Use CPU for deterministic testing or CUDA if available
    device = 'cpu' 
    torch.manual_seed(1337)
    
    # Tiny model for fast testing
    config = GPTConfig(block_size=64, n_layer=2, n_head=2, n_embd=32, vocab_size=100)
    model = GPTLanguageModel(config).to(device)
    model.eval()

    # Dummy input
    idx = torch.randint(0, config.vocab_size, (1, 5)).to(device)
    max_new_tokens = 50

    # 1. Generation WITH KV Cache (Default in model.generate)
    start_time = time.time()
    torch.manual_seed(42) # Reset seed for reproducibility
    out_cached = model.generate(idx, max_new_tokens)
    end_time = time.time()
    time_cached = end_time - start_time
    print(f"Time with KV Cache: {time_cached:.4f}s")

    # 2. Generation WITHOUT KV Cache (Manual loop)
    start_time = time.time()
    torch.manual_seed(42) # Reset seed for reproducibility
    
    # Manual generation loop without cache
    idx_no_cache = idx.clone()
    for _ in range(max_new_tokens):
        # Crop to block size
        idx_cond = idx_no_cache[:, -config.block_size:]
        
        # Forward pass WITHOUT cache
        # We invoke model forward with passed None for past_key_values explicitly (default)
        # It returns (logits, loss, present_kv)
        logits, _, _ = model(idx_cond, past_key_values=None) 
        
        # Get last token prediction
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx_no_cache = torch.cat((idx_no_cache, idx_next), dim=1)
        
    end_time = time.time()
    time_no_cache = end_time - start_time
    print(f"Time without KV Cache: {time_no_cache:.4f}s")
    
    # Verification
    # Compare only the generated tokens (ignoring potential differences in full sequence construction if any)
    match = torch.all(out_cached == idx_no_cache)
    
    if match:
        print("SUCCESS: Outputs match!")
    else:
        print("FAILURE: Outputs do not match!")
        print("Cached:", out_cached)
        print("NoCache:", idx_no_cache)
        
    # Speedup might be small for such a tiny model/sequence, but should exist or be close.
    print(f"Speedup: {time_no_cache / time_cached:.2f}x")

if __name__ == "__main__":
    test_kv_cache()
