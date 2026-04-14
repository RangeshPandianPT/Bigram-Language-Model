import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import argparse

from scripts._bootstrap import ROOT_DIR
from llm.tokenizer import BPETokenizer
from llm.config import GPTConfig, TrainConfig
from llm.model import GPTLanguageModel
from llm.paths import MODEL_PATH, TOKENIZER_PREFIX, ensure_project_dirs

class DPOConfig(TrainConfig):
    beta: float = 0.1  # Temperature parameter for DPO
    dpo_epochs: int = 2
    batch_size: int = 4 # Smaller batch size for DPO as we track two models

def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100):
    """
    Computes the log probabilities of the labels under the given logits.
    logits: (B, T, V)
    labels: (B, T)
    """
    assert logits.shape[:-1] == labels.shape

    # shift labels to match prediction (next-token prediction)
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    
    loss_mask = (labels != ignore_index)
    
    # dummy token to prevent gather errors. Will be masked out anyway.
    labels[labels == ignore_index] = 0
    
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)

def compute_dpo_loss(
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    ref_chosen_logits: torch.Tensor,
    ref_rejected_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float
):
    """
    Computes the Direct Preference Optimization (DPO) Loss.
    """
    policy_chosen_logps = get_batch_logps(policy_chosen_logits, chosen_labels)
    policy_rejected_logps = get_batch_logps(policy_rejected_logits, rejected_labels)
    
    ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_labels)
    ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_labels)
    
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    
    # Compute rewards for logging
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()
    
    return loss, chosen_rewards.mean(), rejected_rewards.mean(), reward_accuracies

def generate_mock_preference_data(tokenizer, num_samples=100, max_len=64):
    """
    Generates dummy preference dataset for demonstration.
    Returns: chosen_inputs, chosen_labels, rejected_inputs, rejected_labels
    """
    print("Generating mock preference data...")
    # In practice, you would load Anthropic/hh-rlhf or a similar dataset
    data = []
    
    prompt_text = "The quick brown fox"
    chosen_resp = " jumps over the lazy dog completely avoiding the sleeping cat."
    rejected_resp = " runs away fast and gets lost in the giant dark forest."
    
    p_ids = tokenizer.encode(prompt_text)
    c_ids = tokenizer.encode(chosen_resp)
    r_ids = tokenizer.encode(rejected_resp)
    
    # Format: [prompt_ids] + [response_ids]
    # Labels mask the prompt_ids by setting them to -100
    for _ in range(num_samples):
        chosen_seq = p_ids + c_ids
        chosen_lab = [-100] * len(p_ids) + c_ids
        
        rejected_seq = p_ids + r_ids
        rejected_lab = [-100] * len(p_ids) + r_ids
        
        # Pad to max_len
        if len(chosen_seq) < max_len:
            chosen_seq += [0] * (max_len - len(chosen_seq))
            chosen_lab += [-100] * (max_len - len(chosen_lab))
            
        if len(rejected_seq) < max_len:
            rejected_seq += [0] * (max_len - len(rejected_seq))
            rejected_lab += [-100] * (max_len - len(rejected_lab))
            
        data.append({
            'chosen_x': torch.tensor(chosen_seq[:max_len], dtype=torch.long),
            'chosen_y': torch.tensor(chosen_lab[:max_len], dtype=torch.long),
            'rejected_x': torch.tensor(rejected_seq[:max_len], dtype=torch.long),
            'rejected_y': torch.tensor(rejected_lab[:max_len], dtype=torch.long),
        })
    return data

def train_dpo():
    ensure_project_dirs()
    
    config = DPOConfig()
    gpt_config = GPTConfig()
    
    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PREFIX))
    gpt_config.vocab_size = len(tokenizer.vocab)
    
    print("\n--- Direct Preference Optimization (DPO) Alignment ---")
    
    # Load Policy Model (trainable)
    policy_model = GPTLanguageModel(gpt_config)
    try:
        policy_model.load_state_dict(torch.load(MODEL_PATH, map_location=config.device))
        print(f"Loaded policy model from {MODEL_PATH}")
    except FileNotFoundError:
        print("Warning: No base model found. Initializing random policy model for demonstration.")
    
    policy_model.to(config.device)
    policy_model.train()
    
    # Load Reference Model (frozen)
    ref_model = GPTLanguageModel(gpt_config)
    try:
        ref_model.load_state_dict(torch.load(MODEL_PATH, map_location=config.device))
        print(f"Loaded reference model from {MODEL_PATH}")
    except FileNotFoundError:
        print("Warning: No base model found. Initializing random reference model for demonstration.")
        
    ref_model.to(config.device)
    ref_model.eval() # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False
        
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-5, weight_decay=0.0)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    
    # Get preference data
    pref_data = generate_mock_preference_data(tokenizer, num_samples=200, max_len=config.block_size)
    
    for epoch in range(config.dpo_epochs):
        np.random.shuffle(pref_data)
        
        for i in range(0, len(pref_data), config.batch_size):
            batch = pref_data[i:i+config.batch_size]
            
            chosen_x = torch.stack([item['chosen_x'] for item in batch]).to(config.device)
            chosen_y = torch.stack([item['chosen_y'] for item in batch]).to(config.device)
            rejected_x = torch.stack([item['rejected_x'] for item in batch]).to(config.device)
            rejected_y = torch.stack([item['rejected_y'] for item in batch]).to(config.device)
            
            # Forward passes
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                # Policy model
                pi_chosen_logits, _, _ = policy_model(chosen_x)
                pi_rejected_logits, _, _ = policy_model(rejected_x)
                
                # Reference model (no gradients needed)
                with torch.no_grad():
                    ref_chosen_logits, _, _ = ref_model(chosen_x)
                    ref_rejected_logits, _, _ = ref_model(rejected_x)
                    
                # Calculate DPO loss
                loss, chosen_rew, rejected_rew, accuracy = compute_dpo_loss(
                    pi_chosen_logits, pi_rejected_logits,
                    ref_chosen_logits, ref_rejected_logits,
                    chosen_y, rejected_y,
                    config.beta
                )
            
            # Optimization step
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if i % (config.batch_size * 5) == 0:
                print(f"Epoch {epoch+1} | Step {i//config.batch_size:03d} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Chosen Rew: {chosen_rew.item():.4f} | "
                      f"Rej Rew: {rejected_rew.item():.4f} | "
                      f"Acc: {accuracy.item():.2f}")

    print("\nDPO Training Complete!")
    dpo_model_path = MODEL_PATH.parent / "model_dpo_aligned.pth"
    torch.save(policy_model.state_dict(), dpo_model_path)
    print(f"Saved DPO-aligned model to {dpo_model_path}")

if __name__ == "__main__":
    train_dpo()