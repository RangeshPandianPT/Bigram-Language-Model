import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeculativeDecoder:
    """
    Speculative Decoding Engine for LLM acceleration.
    Uses a smaller draft model to generate K token candidates, which are then
    validated in parallel by the target model via rejection sampling.
    """
    def __init__(self, target_model: nn.Module, draft_model: nn.Module, gamma: int = 4):
        self.target_model = target_model
        self.draft_model = draft_model
        self.gamma = gamma

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        """
        Generate tokens using speculative decoding.
        prompt_ids: tensor of shape (1, seq_len)
        """
        device = prompt_ids.device
        generated = prompt_ids.clone()
        
        self.target_model.eval()
        self.draft_model.eval()

        tokens_generated = 0
        while tokens_generated < max_new_tokens:
            num_draft = min(self.gamma, max_new_tokens - tokens_generated)
            if num_draft <= 0:
                break

            draft_input = generated.clone()
            draft_probs_list = []
            draft_tokens = []

            # Step 1: Draft model generates K tokens sequentially
            for _ in range(num_draft):
                logits, _, _ = self.draft_model(draft_input) # (1, T, V)
                logits = logits[:, -1, :] # (1, V)
                if temperature > 0:
                    logits = logits / temperature
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')
                    probs = F.softmax(logits, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_tok = torch.argmax(logits, dim=-1, keepdim=True)

                draft_probs_list.append(probs)
                draft_tokens.append(next_tok)
                draft_input = torch.cat([draft_input, next_tok], dim=1)

            draft_tokens_tensor = torch.cat(draft_tokens, dim=1) # (1, K)
            
            # Step 2: Target model runs parallel forward pass over prefix + draft tokens
            target_input = torch.cat([generated, draft_tokens_tensor], dim=1)
            target_logits, _, _ = self.target_model(target_input)
            target_logits = target_logits[:, -(num_draft + 1):, :] # (1, K+1, V)

            if temperature > 0:
                target_logits = target_logits / temperature
            target_probs = F.softmax(target_logits, dim=-1) # (1, K+1, V)

            # Step 3: Rejection sampling verification
            n_accepted = 0
            for i in range(num_draft):
                cand_tok = draft_tokens_tensor[0, i].item()
                p_draft = draft_probs_list[i][0, cand_tok].item()
                p_target = target_probs[0, i, cand_tok].item()

                if p_draft <= 0:
                    r = 1.0
                else:
                    r = min(1.0, p_target / p_draft)

                u = torch.rand(1, device=device).item()
                if u < r:
                    n_accepted += 1
                else:
                    break

            # Accept tokens
            if n_accepted > 0:
                generated = torch.cat([generated, draft_tokens_tensor[:, :n_accepted]], dim=1)
                tokens_generated += n_accepted

            # Step 4: Resample next token if not all draft tokens accepted
            if n_accepted < num_draft and tokens_generated < max_new_tokens:
                p_target_i = target_probs[:, n_accepted, :]
                p_draft_i = draft_probs_list[n_accepted]
                adjusted_probs = torch.clamp(p_target_i - p_draft_i, min=0.0)
                sum_adj = adjusted_probs.sum(dim=-1, keepdim=True)
                if sum_adj.item() > 0:
                    adjusted_probs = adjusted_probs / sum_adj
                    resampled_tok = torch.multinomial(adjusted_probs, num_samples=1)
                else:
                    resampled_tok = torch.multinomial(p_target_i, num_samples=1)
                generated = torch.cat([generated, resampled_tok], dim=1)
                tokens_generated += 1
            elif n_accepted == num_draft and tokens_generated < max_new_tokens:
                # Bonus token from target model
                next_prob = target_probs[:, num_draft, :]
                bonus_tok = torch.multinomial(next_prob, num_samples=1) if temperature > 0 else torch.argmax(next_prob, dim=-1, keepdim=True)
                generated = torch.cat([generated, bonus_tok], dim=1)
                tokens_generated += 1

        return generated
