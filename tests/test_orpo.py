import torch
from src.llm.config import GPTConfig
from src.llm.model import GPTLanguageModel
from train_orpo import compute_sequence_log_probs_and_ce, compute_orpo_loss, run_orpo_training_step


def test_orpo_sequence_log_probs_and_ce():
    config = GPTConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=50, block_size=16)
    model = GPTLanguageModel(config)
    model.eval()

    input_ids = torch.randint(0, 50, (2, 8))
    target_ids = input_ids.clone()
    target_ids[:, :4] = -100 # Mask prompt tokens

    with torch.no_grad():
        avg_log_prob, sft_loss = compute_sequence_log_probs_and_ce(model, input_ids, target_ids)

    assert avg_log_prob.shape == (2,)
    assert torch.all(avg_log_prob <= 0.0) # Log probabilities must be <= 0
    assert sft_loss > 0.0


def test_orpo_loss_computation_and_no_ref_model():
    config = GPTConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=50, block_size=16)
    model = GPTLanguageModel(config)
    model.eval()

    chosen_ids = torch.randint(0, 50, (2, 8))
    chosen_targets = chosen_ids.clone()
    chosen_targets[:, :4] = -100

    rejected_ids = torch.randint(0, 50, (2, 8))
    rejected_targets = rejected_ids.clone()
    rejected_targets[:, :4] = -100

    loss, sft_loss, ratio_loss, log_odds_ratio = compute_orpo_loss(
        model, chosen_ids, chosen_targets, rejected_ids, rejected_targets, beta=0.1
    )

    assert torch.isfinite(loss)
    assert sft_loss > 0.0
    assert ratio_loss >= 0.0


def test_orpo_training_step():
    config = GPTConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=50, block_size=16)
    model = GPTLanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    chosen_ids = torch.randint(0, 50, (2, 8))
    chosen_targets = chosen_ids.clone()
    chosen_targets[:, :4] = -100

    rejected_ids = torch.randint(0, 50, (2, 8))
    rejected_targets = rejected_ids.clone()
    rejected_targets[:, :4] = -100

    metrics = run_orpo_training_step(
        model, optimizer, chosen_ids, chosen_targets, rejected_ids, rejected_targets, beta=0.1
    )

    assert "loss" in metrics
    assert "sft_loss" in metrics
    assert "ratio_loss" in metrics
    assert "log_odds_ratio" in metrics
    assert metrics["loss"] > 0.0
