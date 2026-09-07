import torch
from src.llm.config import GPTConfig
from src.llm.model import GPTLanguageModel, CausalSelfAttention

def test_mla_attention_shape_and_cache():
    config = GPTConfig(
        n_layer=2,
        n_head=4,
        n_embd=64,
        vocab_size=100,
        block_size=32,
        use_mla=True,
        kv_lora_rank=16,
        q_lora_rank=32,
        rope_head_dim=8
    )
    model = GPTLanguageModel(config)
    model.eval()

    idx = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        logits, loss, past_kv = model(idx)

    assert logits.shape == (2, 8, 100)
    assert len(past_kv) == 2

    # Step-by-step cached inference
    next_tok = torch.randint(0, 100, (2, 1))
    with torch.no_grad():
        cached_logits, _, new_kv = model(next_tok, past_key_values=past_kv)

    assert cached_logits.shape == (2, 1, 100)
    assert new_kv[0][0].shape[1] == 9 # 8 past + 1 new
