import torch
import time
import argparse

from scripts._bootstrap import ROOT_DIR
from llm.tokenizer import BPETokenizer
from llm.config import GPTConfig, TrainConfig
from llm.model import GPTLanguageModel
from llm.paths import MODEL_PATH, TOKENIZER_PREFIX

def speculative_decode(target_model, draft_model, context, tokenizer, max_new_tokens=500, gamma=4, temperature=1.0):
    """
    Speculative Decoding loop.
    Target Model: Large, accurate model (Slow autoregressively)
    Draft Model: Small, fast model (Fast autoregressively)
    """
    total_tokens = 0
    accepted_tokens = 0
    
    b, t = context.size()
    seq = context.clone()
    
    start_time = time.time()
    
    with torch.no_grad():
        while seq.shape[1] < t + max_new_tokens:
            past_len = seq.shape[1]
            
            # 1. Draft phase: generate standard batch of 'gamma' tokens sequentially with the small model
            draft_seq = seq.clone()
            
            for _ in range(gamma):
                logits, _, _ = draft_model(draft_seq)
                next_token_logits = logits[:, -1, :] / temperature
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                draft_seq = torch.cat((draft_seq, next_token), dim=1)
                
            draft_tokens = draft_seq[:, past_len:] # The 'gamma' drafted tokens
            
            # 2. Target phase: evaluate the draft tokens in parallel using the massive target model
            # We do a SINGLE forward pass for all drafted tokens
            # Target sees the prefix + 'gamma' draft tokens
            target_logits, _, _ = target_model(draft_seq)
            
            # 3. Acceptance evaluation
            # Compare the draft tokens against what the target model outputs.
            # (Simplification: Greedy exact matching. In reality, we sample and use rejection sampling ratios)
            
            draft_accepted = 0
            n = 0
            
            for i in range(gamma):
                target_token_logits = target_logits[:, past_len - 1 + i, :] / temperature
                target_token = torch.argmax(target_token_logits, dim=-1).unsqueeze(-1)
                draft_token = draft_tokens[:, i].unsqueeze(-1)
                
                if torch.equal(draft_token, target_token):
                    draft_accepted += 1
                else:
                    break
            
            accepted_tokens += draft_accepted
            
            # Calculate the corrected sequence to resume from
            # We keep the correct drafts, and then output what the target model chose at the first point of divergence
            valid_drafts = draft_tokens[:, :draft_accepted]
            target_correction_logits = target_logits[:, past_len - 1 + draft_accepted, :] / temperature
            target_correction = torch.argmax(target_correction_logits, dim=-1).unsqueeze(-1)
            
            seq = torch.cat((seq, valid_drafts, target_correction), dim=1)
            total_tokens += (draft_accepted + 1)
            
            # Yield the accepted output tokens incrementally (for streaming)
            # yield seq[0, past_len:].tolist()

            if seq.shape[1] >= t + max_new_tokens:
                break

    end_time = time.time()
    latency = end_time - start_time
    print(f"\n[Speculative Stats]")
    print(f"Total time   : {latency:.2f} s")
    print(f"Total tokens : {total_tokens}")
    print(f"T/s          : {total_tokens / latency:.2f} t/s")
    print(f"Acceptance % : {accepted_tokens / total_tokens * 100:.1f}%\n")
    
    return seq

def generate():
    # Setup
    train_config = TrainConfig()
    target_config = GPTConfig()
    draft_config = GPTConfig(n_layer=2, n_embd=64, n_head=2) # Tiny 2-layer draft model
    
    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PREFIX))
    target_config.vocab_size = len(tokenizer.vocab)
    draft_config.vocab_size = len(tokenizer.vocab)
    
    print("Loading Models...")
    
    # Init target model
    target_model = GPTLanguageModel(target_config).to(train_config.device)
    try:
        target_model.load_state_dict(torch.load(MODEL_PATH, map_location=train_config.device))
        print("Target model loaded successfully.")
    except FileNotFoundError:
        print("Warning: Target model weights not found. Using untrained weights.")
    target_model.eval()
    
    # Init draft model
    draft_model = GPTLanguageModel(draft_config).to(train_config.device)
    # Note: In practice, you'd load a train draft model here:
    # draft_model.load_state_dict(torch.load(DRAFT_MODEL_PATH))
    draft_model.eval()
    
    prompt = "The future of artificial intelligence is"
    prompt_ids = tokenizer.encode(prompt)
    context = torch.tensor([prompt_ids], dtype=torch.long, device=train_config.device)
    
    print("-" * 80)
    print(f"Prompt: {prompt}")
    print("-" * 80)
    
    # Run standard autoregressive decode (for benchmark comparison)
    print("\n--- STANDARD AUTOREGRESSIVE GENERATION ---")
    t0 = time.time()
    standard_out = target_model.generate(context, max_new_tokens=50)[0]
    t1 = time.time()
    print(tokenizer.decode(standard_out.tolist()))
    print(f"Standard T/s: {50 / (t1 - t0):.2f} t/s")

    # Run Speculative Decode
    print("\n--- SPECULATIVE DECODING ---")
    speculative_out = speculative_decode(
        target_model, draft_model, context, tokenizer, 
        max_new_tokens=50, gamma=4
    )[0]
    
    print(tokenizer.decode(speculative_out.tolist()))

if __name__ == "__main__":
    generate()