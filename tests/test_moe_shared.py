import torch
from src.llm.config import GPTConfig
from src.llm.model import GPTLanguageModel

def test_fine_grained_moe_with_shared_experts():
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=50,
        block_size=16,
        n_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1
    )
    model = GPTLanguageModel(config)
    model.eval()

    x = torch.randint(0, 50, (2, 8))
    with torch.no_grad():
        logits, _, _ = model(x)

    assert logits.shape == (2, 8, 50)
