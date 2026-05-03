import torch
import torch.nn.functional as F

@torch.no_grad()
def get_choice_log_likelihood(model, tokenizer, context: str, choice: str, device: str) -> float:
    """
    Computes the log-likelihood of a specific choice (continuation) given a context.
    """
    # Tokenize context and the full text (context + choice)
    ctx_ids = tokenizer.encode(context)
    full_text = context + choice
    full_ids = tokenizer.encode(full_text)
    
    # We only care about the predictions for the continuation.
    continuation_ids = full_ids[len(ctx_ids):]
    
    if len(continuation_ids) == 0:
        return -float('inf')
        
    block_size = getattr(model.config, 'block_size', 128)
    
    # If the sequence is too long for the model's context window, truncate from the left
    # But ensure we don't truncate the continuation itself (or at least keep as much as possible)
    if len(full_ids) > block_size:
        truncate_len = len(full_ids) - block_size
        # Ideally we only truncate context. If continuation is longer than block_size, we just take the last block_size tokens.
        full_ids = full_ids[truncate_len:]
        # Update the context boundary
        ctx_len = len(ctx_ids) - truncate_len
        if ctx_len < 0:
            ctx_len = 0 # Continuation itself is longer than block_size
    else:
        ctx_len = len(ctx_ids)
        
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    
    # model returns logits shape (1, T, vocab_size)
    logits, _, _ = model(input_ids)
    
    # We only care about the predictions for the continuation.
    # The first token of continuation is predicted by the last token of context.
    # The index in full_ids of the last context token is ctx_len - 1.
    start_idx = max(0, ctx_len - 1)
    end_idx = len(full_ids) - 1
    
    # logits to consider: from start_idx to end_idx-1
    relevant_logits = logits[0, start_idx:end_idx, :] # shape (len(continuation_ids), vocab_size)
    
    log_probs = F.log_softmax(relevant_logits, dim=-1)
    
    # target tokens
    target_ids = torch.tensor(continuation_ids, dtype=torch.long, device=device)
    
    # gather the log probabilities of the target tokens
    choice_log_probs = log_probs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
    
    return choice_log_probs.sum().item()
