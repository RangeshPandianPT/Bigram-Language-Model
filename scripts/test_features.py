import torch
import torch.nn as nn
from llm.config import GPTConfig, TrainConfig
from llm.model import GPTLanguageModel

def test_weight_tying():
    print("Testing Weight Tying...")
    # Without weight tying
    config1 = GPTConfig(vocab_size=1000, n_embd=128, tie_word_embeddings=False)
    model1 = GPTLanguageModel(config1)
    params1 = sum(p.numel() for p in model1.parameters())
    
    # With weight tying
    config2 = GPTConfig(vocab_size=1000, n_embd=128, tie_word_embeddings=True)
    model2 = GPTLanguageModel(config2)
    params2 = sum(p.numel() for p in model2.parameters())
    
    print(f"Params (Untied): {params1}")
    print(f"Params (Tied): {params2}")
    
    assert params2 < params1, "Tied model should have fewer parameters!"
    assert torch.equal(model2.lm_head.weight, model2.token_embedding_table.weight), "Weights are not tied!"
    print("Weight tying test passed!\n")

def test_min_p():
    print("Testing Min-P Sampling...")
    config = GPTConfig(vocab_size=1000, n_embd=128)
    model = GPTLanguageModel(config)
    
    # Dummy input
    idx = torch.randint(0, 1000, (1, 10))
    
    # Generate 5 tokens with min_p
    out = model.generate(idx, max_new_tokens=5, min_p=0.1)
    
    assert out.shape == (1, 15), "Output shape is incorrect!"
    print("Min-P sampling test passed!\n")

def test_gradient_checkpointing():
    print("Testing Gradient Checkpointing...")
    config = GPTConfig(vocab_size=1000, n_embd=128)
    config.gradient_checkpointing = True
    model = GPTLanguageModel(config)
    model.train() # Must be in training mode
    
    x = torch.randint(0, 1000, (2, 32))
    y = torch.randint(0, 1000, (2, 32))
    
    # Forward pass
    logits, loss, _ = model(x, targets=y)
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist
    assert model.token_embedding_table.weight.grad is not None, "Gradients not computed!"
    print("Gradient Checkpointing test passed!\n")

if __name__ == "__main__":
    test_weight_tying()
    test_min_p()
    test_gradient_checkpointing()
    print("All feature tests passed successfully!")
