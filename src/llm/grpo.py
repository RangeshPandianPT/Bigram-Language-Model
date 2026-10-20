import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


def extract_tag_content(text: str, tag: str) -> Optional[str]:
    """
    Extracts text between opening and closing XML-style tags, e.g. <think>...</think>.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def format_reward_func(completions: List[str]) -> torch.Tensor:
    """
    Reward function that verifies presence and order of reasoning (<think>) and answer (<answer>) tags.
    - 1.0: Full structure (<think>...</think><answer>...</answer>)
    - 0.5: Has <think> or <answer> tags but not both
    - 0.0: No structured reasoning tags
    """
    rewards = []
    for comp in completions:
        has_think = ("<think>" in comp) and ("</think>" in comp)
        has_answer = ("<answer>" in comp) and ("</answer>" in comp)
        
        if has_think and has_answer:
            # Check ordering: think must come before answer
            think_end = comp.find("</think>")
            ans_start = comp.find("<answer>")
            if think_end < ans_start:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        elif has_think or has_answer:
            rewards.append(0.3)
        else:
            rewards.append(0.0)
            
    return torch.tensor(rewards, dtype=torch.float32)


def accuracy_reward_func(completions: List[str], ground_truths: List[str]) -> torch.Tensor:
    """
    Reward function that compares the extracted <answer>...</answer> against ground truth.
    Supports exact string match or normalized numerical match.
    """
    rewards = []
    for comp, truth in zip(completions, ground_truths):
        extracted = extract_tag_content(comp, "answer")
        if extracted is None:
            # Fallback: check last non-empty line
            lines = [l.strip() for l in comp.strip().splitlines() if l.strip()]
            extracted = lines[-1] if lines else ""
            
        extracted_clean = extracted.strip().lower()
        truth_clean = truth.strip().lower()
        
        if extracted_clean == truth_clean:
            rewards.append(1.0)
        else:
            # Try float comparison if both are numeric
            try:
                if abs(float(extracted_clean) - float(truth_clean)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except ValueError:
                rewards.append(0.0)
                
    return torch.tensor(rewards, dtype=torch.float32)


def reasoning_length_reward_func(
    completions: List[str],
    min_tokens: int = 5,
    max_tokens: int = 200
) -> torch.Tensor:
    """
    Penalizes empty or degenerate thinking while rewarding thorough thought processes.
    """
    rewards = []
    for comp in completions:
        think = extract_tag_content(comp, "think")
        if think is None:
            rewards.append(0.0)
        else:
            words = len(think.split())
            if words < min_tokens:
                rewards.append(0.2)
            elif words <= max_tokens:
                rewards.append(1.0)
            else:
                # Slight penalty for exceeding length limit
                rewards.append(0.5)
    return torch.tensor(rewards, dtype=torch.float32)


def compute_group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Computes group-relative normalized advantages for GRPO:
    A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
    
    Args:
        rewards: Tensor of shape (B * group_size,) or (B, group_size)
        group_size: Number of completions sampled per prompt
        eps: Small constant for numerical stability
        
    Returns:
        Tensor of shape (B * group_size,) representing normalized advantages
    """
    flat_rewards = rewards.view(-1)
    num_samples = flat_rewards.size(0)
    assert num_samples % group_size == 0, f"Total samples ({num_samples}) must be divisible by group_size ({group_size})"
    
    batch_size = num_samples // group_size
    rewards_grouped = flat_rewards.view(batch_size, group_size)
    
    mean = rewards_grouped.mean(dim=1, keepdim=True)
    std = rewards_grouped.std(dim=1, keepdim=True)
    
    # When std is zero (all rewards identical in group), advantage is 0
    advantages = (rewards_grouped - mean) / (std + eps)
    advantages = torch.where(std < 1e-7, torch.zeros_like(advantages), advantages)
    
    return advantages.view(-1)


def get_per_token_logps(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor
) -> torch.Tensor:
    """
    Computes per-token log probabilities for causal LM.
    
    Args:
        model: Language model
        input_ids: (N, T) input token IDs
        target_ids: (N, T) target token IDs with -100 for prompt tokens
        
    Returns:
        Tensor of shape (N, T-1) with per-token log probabilities (0.0 for masked tokens)
    """
    logits, _, _ = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1) # (N, T, V)
    
    shift_log_probs = log_probs[:, :-1, :] # (N, T-1, V)
    shift_targets = target_ids[:, 1:]      # (N, T-1)
    
    mask = (shift_targets != -100)
    safe_targets = shift_targets.clone()
    safe_targets[~mask] = 0
    
    per_token_logps = torch.gather(shift_log_probs, dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    per_token_logps = per_token_logps * mask.float()
    return per_token_logps


def compute_grpo_loss(
    model: nn.Module,
    ref_model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    advantages: torch.Tensor,
    old_logps: Optional[torch.Tensor] = None,
    clip_eps: float = 0.2,
    beta: float = 0.04
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes Group Relative Policy Optimization (GRPO) clipped surrogate loss
    and per-token KL divergence penalty with reference model.
    
    Args:
        model: Policy model being optimized
        ref_model: Frozen reference model
        input_ids: (B*G, T) sequence input IDs
        target_ids: (B*G, T) target sequence IDs (-100 for prompt tokens)
        advantages: (B*G,) group normalized advantages
        old_logps: (B*G, T-1) log probabilities under behavior policy (or None)
        clip_eps: PPO clipping epsilon
        beta: KL penalty coefficient
        
    Returns:
        loss: Scalar loss tensor with gradients
        metrics: Dictionary of diagnostic metrics
    """
    mask = (target_ids[:, 1:] != -100).float() # (N, T-1)
    
    # 1. Current policy log probabilities
    current_logps = get_per_token_logps(model, input_ids, target_ids) # (N, T-1)
    
    # 2. Old policy log probabilities (if None, use detached current)
    if old_logps is None:
        old_logps = current_logps.detach()
        
    # 3. Reference policy log probabilities
    with torch.no_grad():
        ref_logps = get_per_token_logps(ref_model, input_ids, target_ids)
        
    # 4. Probability ratio r_t = pi_theta(t) / pi_old(t)
    log_ratio = current_logps - old_logps
    ratio = torch.exp(log_ratio) # (N, T-1)
    
    # 5. Clipped surrogate loss per token
    adv_expanded = advantages.unsqueeze(1).expand_as(ratio) # (N, T-1)
    surr1 = ratio * adv_expanded
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_expanded
    policy_loss_per_token = -torch.min(surr1, surr2) * mask
    
    # Average policy loss over active completion tokens
    token_counts = mask.sum(dim=1).clamp(min=1.0)
    per_seq_policy_loss = policy_loss_per_token.sum(dim=1) / token_counts
    policy_loss = per_seq_policy_loss.mean()
    
    # 6. Unbiased Schulman KL divergence: D_KL = exp(log(ref) - log(pi)) - (log(ref) - log(pi)) - 1
    log_ratio_ref = ref_logps - current_logps
    kl_per_token = (torch.exp(log_ratio_ref) - log_ratio_ref - 1.0) * mask
    per_seq_kl = kl_per_token.sum(dim=1) / token_counts
    kl_loss = per_seq_kl.mean()
    
    # Total GRPO objective: minimize policy loss + beta * KL
    total_loss = policy_loss + beta * kl_loss
    
    metrics = {
        "loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "kl_div": kl_loss.item(),
        "mean_ratio": (ratio * mask).sum().item() / mask.sum().clamp(min=1).item(),
        "mean_advantage": advantages.mean().item()
    }
    
    return total_loss, metrics
