import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from src.llm.model import GPTLanguageModel
from src.llm.config import GPTConfig, TrainConfig

def compute_sequence_logp(model: nn.Module, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """
    Computes log p_\theta(target_ids | input_ids) summed over non-masked target tokens (-100 ignored).
    input_ids: (B, T)
    target_ids: (B, T)
    """
    logits, _, _ = model(input_ids) # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1) # (B, T, V)

    # Shift logits and targets for causal next-token log probability
    shift_log_probs = log_probs[:, :-1, :] # (B, T-1, V)
    shift_targets = target_ids[:, 1:]      # (B, T-1)

    mask = (shift_targets != -100)
    safe_targets = shift_targets.clone()
    safe_targets[~mask] = 0

    per_token_logps = torch.gather(shift_log_probs, dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1) # (B, T-1)
    per_token_logps = per_token_logps * mask.float()

    return per_token_logps.sum(dim=-1) # (B,)


def compute_dpo_loss(
    policy_model: nn.Module,
    ref_model: nn.Module,
    chosen_ids: torch.Tensor,
    chosen_targets: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_targets: torch.Tensor,
    beta: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Direct Preference Optimization (DPO) Loss function.
    """
    policy_chosen_logp = compute_sequence_logp(policy_model, chosen_ids, chosen_targets)
    policy_rejected_logp = compute_sequence_logp(policy_model, rejected_ids, rejected_targets)

    with torch.no_grad():
        ref_chosen_logp = compute_sequence_logp(ref_model, chosen_ids, chosen_targets)
        ref_rejected_logp = compute_sequence_logp(ref_model, rejected_ids, rejected_targets)

    pi_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp

    logits = beta * (pi_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()

    chosen_rewards = beta * (policy_chosen_logp - ref_chosen_logp).detach()
    rejected_rewards = beta * (policy_rejected_logp - ref_rejected_logp).detach()

    return loss, chosen_rewards.mean(), rejected_rewards.mean()


def run_dpo_training_step(
    policy_model: nn.Module,
    ref_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    chosen_ids: torch.Tensor,
    chosen_targets: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_targets: torch.Tensor,
    beta: float = 0.1
) -> dict:
    policy_model.train()
    optimizer.zero_grad()

    loss, chosen_reward, rejected_reward = compute_dpo_loss(
        policy_model, ref_model, chosen_ids, chosen_targets, rejected_ids, rejected_targets, beta=beta
    )
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "chosen_reward": chosen_reward.item(),
        "rejected_reward": rejected_reward.item()
    }


if __name__ == "__main__":
    print("DPO Trainer initialized successfully.")
    config = GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=100, block_size=32)
    policy_model = GPTLanguageModel(config)
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-4)

    # Synthetic preference batch
    chosen_ids = torch.randint(0, 100, (2, 16))
    chosen_targets = chosen_ids.clone()
    chosen_targets[:, :8] = -100

    rejected_ids = torch.randint(0, 100, (2, 16))
    rejected_targets = rejected_ids.clone()
    rejected_targets[:, :8] = -100

    metrics = run_dpo_training_step(
        policy_model, ref_model, optimizer, chosen_ids, chosen_targets, rejected_ids, rejected_targets
    )
    print("DPO Training Step Metrics:", metrics)
