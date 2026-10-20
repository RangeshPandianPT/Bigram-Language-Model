import copy
import torch
import torch.nn as nn
from src.llm.config import GPTConfig
from src.llm.model import GPTLanguageModel
from src.llm.grpo import (
    compute_group_advantages,
    format_reward_func,
    accuracy_reward_func,
    reasoning_length_reward_func,
    compute_grpo_loss,
    get_per_token_logps
)
from train_grpo import run_grpo_training_step, GRPOTrainer


def test_group_advantage_normalization():
    # 2 prompts, 4 rollouts each
    group_size = 4
    # Group 1 has varying rewards, Group 2 has varying rewards
    rewards = torch.tensor([1.0, 0.5, 0.0, 1.5, 2.0, 4.0, 6.0, 8.0])
    adv = compute_group_advantages(rewards, group_size=group_size)
    
    assert adv.shape == (8,)
    # Check mean of group 1 is close to 0
    assert torch.isclose(adv[:group_size].mean(), torch.tensor(0.0), atol=1e-5)
    # Check std of group 1 is close to 1
    assert torch.isclose(adv[:group_size].std(), torch.tensor(1.0), atol=1e-3)
    # Check mean of group 2 is close to 0
    assert torch.isclose(adv[group_size:].mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(adv[group_size:].std(), torch.tensor(1.0), atol=1e-3)
    
    # Edge case: Identical rewards in a group should result in zero advantage
    constant_rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
    const_adv = compute_group_advantages(constant_rewards, group_size=group_size)
    assert torch.all(const_adv == 0.0)


def test_rule_based_reward_functions():
    completions = [
        "<think>Steps: 2 + 2 = 4</think><answer>4</answer>",
        "<answer>4</answer><think>Thinking after answer</think>",
        "<think>Only thinking</think>",
        "Just a plain response without tags"
    ]
    ground_truths = ["4", "4", "4", "4"]
    
    # 1. Format reward
    fmt_rew = format_reward_func(completions)
    assert fmt_rew[0] == 1.0   # valid order
    assert fmt_rew[1] == 0.5   # inverted order
    assert fmt_rew[2] == 0.3   # only think tag
    assert fmt_rew[3] == 0.0   # no tags
    
    # 2. Accuracy reward
    acc_rew = accuracy_reward_func(completions, ground_truths)
    assert acc_rew[0] == 1.0   # matches 4
    assert acc_rew[1] == 1.0   # matches 4
    assert acc_rew[2] == 0.0   # no answer tag
    
    # 3. Numeric accuracy match
    numeric_comp = ["<answer> 4.000 </answer>"]
    numeric_rew = accuracy_reward_func(numeric_comp, ["4"])
    assert numeric_rew[0] == 1.0


def test_grpo_loss_and_gradients():
    config = GPTConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=50, block_size=32)
    policy = GPTLanguageModel(config)
    ref = copy.deepcopy(policy)
    ref.eval()
    
    group_size = 2
    num_rollouts = 4 # 2 prompts * 2 completions
    seq_len = 16
    
    input_ids = torch.randint(0, 50, (num_rollouts, seq_len))
    target_ids = input_ids.clone()
    target_ids[:, :6] = -100 # Mask prompt
    
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
    
    loss, metrics = compute_grpo_loss(
        model=policy,
        ref_model=ref,
        input_ids=input_ids,
        target_ids=target_ids,
        advantages=advantages,
        clip_eps=0.2,
        beta=0.04
    )
    
    assert torch.isfinite(loss)
    assert "policy_loss" in metrics
    assert "kl_div" in metrics
    assert "mean_ratio" in metrics
    
    # Gradient check
    loss.backward()
    has_policy_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in policy.parameters())
    assert has_policy_grad
    # Ensure reference model was not given gradients
    has_ref_grad = any(p.grad is not None for p in ref.parameters())
    assert not has_ref_grad


def test_grpo_trainer_step():
    config = GPTConfig(n_layer=2, n_head=2, n_embd=32, vocab_size=50, block_size=32)
    policy = GPTLanguageModel(config)
    
    trainer = GRPOTrainer(policy_model=policy, group_size=2)
    
    input_ids = torch.randint(0, 50, (4, 16))
    target_ids = input_ids.clone()
    target_ids[:, :6] = -100
    rewards = torch.tensor([1.0, 0.0, 0.5, 0.8])
    
    metrics = trainer.train_step(input_ids, target_ids, rewards)
    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "kl_div" in metrics
