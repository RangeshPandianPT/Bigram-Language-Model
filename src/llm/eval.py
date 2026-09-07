import torch
import torch.nn as nn
import torch.nn.functional as F

class Evaluator:
    """
    Automated LLM Evaluation Harness.
    Performs log-likelihood evaluation for multiple-choice tasks (e.g. MMLU, ARC, PIQA)
    and computes model perplexity.
    """
    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def evaluate_multiple_choice(self, prompt_tokens: list[int], option_tokens_list: list[list[int]]) -> int:
        """
        Calculates normalized log-likelihood for each option given prompt_tokens.
        Returns index of the option with highest normalized log-likelihood.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        best_score = -float('inf')
        best_idx = 0

        for i, option_tokens in enumerate(option_tokens_list):
            full_seq = prompt_tokens + option_tokens
            input_ids = torch.tensor([full_seq], dtype=torch.long, device=device)

            logits = self.model(input_ids) # (1, T, V)
            log_probs = F.log_softmax(logits, dim=-1) # (1, T, V)

            # Sum log probabilities of option tokens
            start_pos = len(prompt_tokens) - 1
            option_len = len(option_tokens)

            option_logp = 0.0
            for pos_idx, tok_id in enumerate(option_tokens):
                curr_pos = start_pos + pos_idx
                tok_logp = log_probs[0, curr_pos, tok_id].item()
                option_logp += tok_logp

            norm_score = option_logp / max(1, option_len)
            if norm_score > best_score:
                best_score = norm_score
                best_idx = i

        return best_idx

    @torch.no_grad()
    def compute_perplexity(self, text_tokens: list[int], stride: int = 128, max_len: int = 256) -> float:
        """
        Computes text perplexity using a sliding window.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        
        nlls = []
        for i in range(0, len(text_tokens) - 1, stride):
            begin_loc = max(i + stride - max_len, 0)
            end_loc = min(i + stride, len(text_tokens))
            trg_len = end_loc - i

            input_ids = torch.tensor([text_tokens[begin_loc:end_loc]], dtype=torch.long, device=device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            logits = self.model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = target_ids[:, 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
            nlls.append(loss * trg_len)

        if not nlls:
            return float('nan')
        return torch.exp(torch.stack(nlls).sum() / (len(text_tokens) - 1)).item()
