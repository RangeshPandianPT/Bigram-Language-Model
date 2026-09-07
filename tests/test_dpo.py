import torch
import copy
from src.llm.config import GPTConfig
from src.llm.model import GPTLanguageModel
from train_dpo import compute_dpo_loss, run_dpo_training_step

def test_dpo_loss_computation_and_gradient():
    config = GPTConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=50, block_size=16)
    policy = GPTLanguageModel(config)
    ref = copy.deepcopy(policy)
    ref.eval()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    chosen_ids = torch.randint(0, 50, (2, 8))
    chosen_targets = chosen_ids.clone()
    chosen_targets[:, :4] = -100

    rejected_ids = torch.randint(0, 50, (2, 8))
    rejected_targets = rejected_ids.clone()
    rejected_targets[:, :4] = -100

    metrics = run_dpo_training_step(policy, ref, optimizer, chosen_ids, chosen_targets, rejected_ids, rejected_targets)

    assert "loss" in metrics
    assert metrics["loss"] > 0.0
