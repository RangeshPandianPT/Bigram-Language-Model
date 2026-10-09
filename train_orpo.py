import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from src.llm.model import GPTLanguageModel
from src.llm.config import GPTConfig, TrainConfig

def compute_sequence_log_probs_and_ce(model: nn.Module, input_ids: torch.Tensor, target_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes average per-token log probabilities and CE SFT loss.
    input_ids: (B, T)
    target_ids: (B, T)
    """
    logits, _, _ = model(input_ids) # (B, T, V)
    
    # Shift logits and targets for causal next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = target_ids[:, 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
    ce_loss = ce_loss.view(shift_targets.size()) # (B, T-1)
    
    mask = (shift_targets != -100)
    
    # log(P(y_t)) = -ce_loss
    log_probs = -ce_loss * mask.float()
    avg_log_prob = log_probs.sum(dim=-1) / mask.float().sum(dim=-1).clamp(min=1) # (B,)
    
    # CE SFT loss (mean over batch and unmasked tokens)
    sft_loss = ce_loss[mask].mean()
    
    return avg_log_prob, sft_loss

def compute_orpo_loss(
    model: nn.Module,
    chosen_ids: torch.Tensor,
    chosen_targets: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_targets: torch.Tensor,
    beta: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Odds Ratio Preference Optimization (ORPO) Loss function.
    ORPO modifies the SFT objective by adding a negative log-odds ratio penalty.
    Crucially, it does NOT require a reference model, saving 50% memory compared to DPO.
    """
    chosen_logp, sft_loss = compute_sequence_log_probs_and_ce(model, chosen_ids, chosen_targets)
    rejected_logp, _ = compute_sequence_log_probs_and_ce(model, rejected_ids, rejected_targets)
    
    # Compute log odds = log P(y) - log(1 - P(y))
    # where P(y) = exp(average_log_prob)
    chosen_log_odds = chosen_logp - torch.log1p(-torch.exp(chosen_logp)) 
    rejected_log_odds = rejected_logp - torch.log1p(-torch.exp(rejected_logp))
    
    # log odds ratio
    log_odds_ratio = chosen_log_odds - rejected_log_odds
    
    # OR loss
    ratio_loss = -F.logsigmoid(log_odds_ratio).mean()
    
    # Total loss: L_SFT + beta * L_OR
    loss = sft_loss + beta * ratio_loss
    
    return loss, sft_loss, ratio_loss, log_odds_ratio.mean()

def run_orpo_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    chosen_ids: torch.Tensor,
    chosen_targets: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_targets: torch.Tensor,
    beta: float = 0.1
) -> dict:
    model.train()
    optimizer.zero_grad()
    
    loss, sft_loss, ratio_loss, log_odds_ratio = compute_orpo_loss(
        model, chosen_ids, chosen_targets, rejected_ids, rejected_targets, beta=beta
    )
    loss.backward()
    optimizer.step()
    
    return {
        "loss": loss.item(),
        "sft_loss": sft_loss.item(),
        "ratio_loss": ratio_loss.item(),
        "log_odds_ratio": log_odds_ratio.item()
    }

if __name__ == "__main__":
    print("ORPO Trainer initialized successfully.")
    config = GPTConfig(n_layer=2, n_head=2, n_embd=64, vocab_size=100, block_size=32)
    model = GPTLanguageModel(config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Synthetic preference batch
    chosen_ids = torch.randint(0, 100, (2, 16))
    chosen_targets = chosen_ids.clone()
    chosen_targets[:, :8] = -100 # Mask prompt
    
    rejected_ids = torch.randint(0, 100, (2, 16))
    rejected_targets = rejected_ids.clone()
    rejected_targets[:, :8] = -100 # Mask prompt
    
    # Notice we only pass one model! No reference model needed.
    metrics = run_orpo_training_step(
        model, optimizer, chosen_ids, chosen_targets, rejected_ids, rejected_targets, beta=0.1
    )
    print("ORPO Training Step Metrics:", metrics)
    print("ORPO test successful! Note how we didn't need a reference model.")
