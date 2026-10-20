import copy
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

from src.llm.model import GPTLanguageModel
from src.llm.config import GPTConfig, TrainConfig
from src.llm.grpo import (
    format_reward_func,
    accuracy_reward_func,
    reasoning_length_reward_func,
    compute_group_advantages,
    compute_grpo_loss
)


class GRPOTrainer:
    """
    Group Relative Policy Optimization (GRPO) Trainer (DeepSeek-R1 style).
    
    Eliminates the separate critic/value model from traditional PPO by:
    1. Sampling a group of G candidate rollouts for each prompt.
    2. Evaluating candidate completions with rule-based reward functions (Format, Accuracy, Length).
    3. Normalizing advantages across each prompt's group: A_i = (R_i - mean(R)) / std(R).
    4. Optimizing the policy via a clipped surrogate objective with an unbiased token-level KL penalty.
    """
    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        group_size: int = 4,
        clip_eps: float = 0.2,
        beta: float = 0.04,
        lr: float = 1e-4,
        reward_weights: Optional[Dict[str, float]] = None
    ):
        self.policy_model = policy_model
        if ref_model is None:
            self.ref_model = copy.deepcopy(policy_model)
        else:
            self.ref_model = ref_model
            
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
            
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.beta = beta
        self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=lr)
        
        self.reward_weights = reward_weights or {
            "format": 0.5,
            "accuracy": 1.0,
            "length": 0.2
        }

    def compute_composite_rewards(
        self,
        completions: List[str],
        ground_truths: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes composite reward scores for completions by evaluating multiple verifiers.
        """
        fmt_rew = format_reward_func(completions)
        acc_rew = accuracy_reward_func(completions, ground_truths)
        len_rew = reasoning_length_reward_func(completions)
        
        total_rewards = (
            self.reward_weights["format"] * fmt_rew +
            self.reward_weights["accuracy"] * acc_rew +
            self.reward_weights["length"] * len_rew
        )
        
        diagnostics = {
            "reward_total_mean": total_rewards.mean().item(),
            "reward_format_mean": fmt_rew.mean().item(),
            "reward_accuracy_mean": acc_rew.mean().item(),
            "reward_length_mean": len_rew.mean().item(),
        }
        return total_rewards, diagnostics

    def train_step(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        rewards: torch.Tensor,
        old_logps: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Executes a single optimization step given inputs, targets, and computed rewards.
        """
        self.policy_model.train()
        self.optimizer.zero_grad()
        
        # 1. Compute group-relative normalized advantages
        advantages = compute_group_advantages(rewards, group_size=self.group_size)
        
        # 2. Compute GRPO surrogate loss with KL divergence
        loss, metrics = compute_grpo_loss(
            model=self.policy_model,
            ref_model=self.ref_model,
            input_ids=input_ids,
            target_ids=target_ids,
            advantages=advantages,
            old_logps=old_logps,
            clip_eps=self.clip_eps,
            beta=self.beta
        )
        
        # 3. Backpropagate and step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return metrics


def run_grpo_training_step(
    policy_model: nn.Module,
    ref_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    rewards: torch.Tensor,
    group_size: int = 4,
    clip_eps: float = 0.2,
    beta: float = 0.04
) -> Dict[str, float]:
    """
    Functional interface to run a single GRPO training step.
    """
    policy_model.train()
    optimizer.zero_grad()
    
    advantages = compute_group_advantages(rewards, group_size=group_size)
    
    loss, metrics = compute_grpo_loss(
        model=policy_model,
        ref_model=ref_model,
        input_ids=input_ids,
        target_ids=target_ids,
        advantages=advantages,
        clip_eps=clip_eps,
        beta=beta
    )
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return metrics


if __name__ == "__main__":
    print("=" * 60)
    print("[GRPO] Initializing DeepSeek-R1 Style GRPO Reasoning Trainer")
    print("=" * 60)
    
    # 1. Setup small model configuration
    config = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=120,
        block_size=64
    )
    policy = GPTLanguageModel(config)
    ref = copy.deepcopy(policy)
    
    group_size = 4
    num_prompts = 2
    total_rollouts = num_prompts * group_size
    seq_len = 32
    prompt_len = 12
    
    # 2. Simulate synthetic rollouts and target masking
    input_ids = torch.randint(0, 120, (total_rollouts, seq_len))
    target_ids = input_ids.clone()
    target_ids[:, :prompt_len] = -100 # Mask prompt tokens
    
    # 3. Mock completions with reasoning patterns for rewards
    sample_completions = [
        "<think>Let's add 15 and 27. 15+20=35, 35+7=42.</think><answer>42</answer>",
        "<think>Calculating sum 15 + 27...</think><answer>42</answer>",
        "<think>Maybe 40?</think><answer>40</answer>",
        "The answer is 42",
        "<think>Multiply 60 by 3: 60 * 3 = 180.</think><answer>180</answer>",
        "<think>Distance is speed times time: 60 * 3.</think><answer>180</answer>",
        "<think>Fast car goes 120 miles.</think><answer>120</answer>",
        "I think it is 180"
    ]
    ground_truths = ["42", "42", "42", "42", "180", "180", "180", "180"]
    
    trainer = GRPOTrainer(policy_model=policy, ref_model=ref, group_size=group_size)
    rewards, rew_diag = trainer.compute_composite_rewards(sample_completions, ground_truths)
    
    print("\nReward Verifier Diagnostics:")
    for k, v in rew_diag.items():
        print(f"  - {k}: {v:.4f}")
        
    print("\nIndividual Rollout Rewards:")
    for i, (comp, rew) in enumerate(zip(sample_completions, rewards)):
        print(f"  Rollout {i+1} [Reward: {rew.item():.2f}]: {comp}")
        
    # 4. Perform GRPO training step
    step_metrics = trainer.train_step(
        input_ids=input_ids,
        target_ids=target_ids,
        rewards=rewards
    )
    
    print("\nGRPO Step Metrics:")
    for k, v in step_metrics.items():
        print(f"  - {k}: {v:.4f}")
        
    print("\n[OK] GRPO Training demonstration completed successfully!")
    print("Notice: No critic/value model required. Normalized group advantages drive reasoning!")
