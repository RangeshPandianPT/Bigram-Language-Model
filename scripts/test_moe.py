import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from llm.config import GPTConfig
from llm.model import GPTLanguageModel

def test_moe():
    torch.manual_seed(42)
    
    print("Testing Mixture of Experts (MoE) Architecture...")
    
    # 1. Test Base Model without MoE
    print("\n1. Instantiating Base Model (No MoE)...")
    base_config = GPTConfig(
         block_size=16,
         vocab_size=100,
         n_layer=2,
         n_head=4,
         n_embd=64,
         n_experts=0
    )
    base_model = GPTLanguageModel(base_config)
    base_params = sum(p.numel() for p in base_model.parameters())
    print(f"   Base Model Parameters: {base_params:,}")
    
    # 2. Test MoE Model with 4 experts
    print("\n2. Instantiating MoE Model (n_experts=4)...")
    moe_config = GPTConfig(
         block_size=16,
         vocab_size=100,
         n_layer=2,
         n_head=4,
         n_embd=64,
         n_experts=4,
         num_experts_per_tok=2
    )
    moe_model = GPTLanguageModel(moe_config)
    moe_params = sum(p.numel() for p in moe_model.parameters())
    print(f"   MoE Model Parameters:  {moe_params:,}")
    
    param_increase = (moe_params / base_params)
    print(f"   Multiplier: {param_increase:.2f}x")
    
    # Assert MoE has significantly more parameters
    assert moe_params > base_params * 1.5, "MoE model should have more parameters due to experts"
    
    # 3. Test Forward Pass
    print("\n3. Testing Forward Pass...")
    x = torch.randint(0, 100, (2, 16)) # Batch=2, Seq=16
    
    # Without past_kv
    logits, loss, past_kv = moe_model(x, targets=x)
    assert logits.shape == (32, 100), f"Expected flattened logits shape (32, 100), got {logits.shape}"
    assert loss is not None, "Loss should not be None when targets are provided"
    
    # 4. Generate integration
    print("4. Testing auto-generate loop (which tests KV cache internally)...")
    print("5. Testing auto-generate loop...")
    out = moe_model.generate(x[:, :4], max_new_tokens=5, temperature=1.0)
    assert out.shape == (2, 9), f"Expected generated shape (2, 9), got {out.shape}"
    
    print("\n[OK] All MoE tests passed Successfully!")

if __name__ == "__main__":
    test_moe()
