import torch
from src.llm.config import GPTConfig
from src.llm.model import GPTLanguageModel
from src.llm.speculative import SpeculativeDecoder

def test_speculative_decoding_generation():
    target_config = GPTConfig(n_layer=4, n_head=4, n_embd=64, vocab_size=100, block_size=64)
    draft_config = GPTConfig(n_layer=1, n_head=2, n_embd=32, vocab_size=100, block_size=64)

    target_model = GPTLanguageModel(target_config)
    draft_model = GPTLanguageModel(draft_config)

    spec_decoder = SpeculativeDecoder(target_model=target_model, draft_model=draft_model, gamma=3)

    prompt_ids = torch.tensor([[1, 5, 10]], dtype=torch.long)
    output_ids = spec_decoder.generate(prompt_ids, max_new_tokens=8, temperature=1.0)

    assert output_ids.shape[0] == 1
    assert output_ids.shape[1] == 3 + 8 # prompt length + max_new_tokens
