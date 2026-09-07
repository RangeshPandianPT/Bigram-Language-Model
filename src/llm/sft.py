import torch
from typing import List, Dict

class PromptFormatter:
    """
    Prompt template formatter supporting ChatML and Llama 3 header styles.
    Creates input token IDs and loss target masks (ignoring prompt tokens with -100).
    """
    def __init__(self, style: str = "llama3"):
        self.style = style

    def format_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: [{'role': 'system'|'user'|'assistant', 'content': '...'}]
        """
        if self.style == "chatml":
            formatted = ""
            for msg in messages:
                formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            return formatted
        else:
            # Default Llama 3 style
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
            return formatted

    def build_sft_targets(self, prompt_tokens: List[int], response_tokens: List[int], ignore_index: int = -100) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Combines prompt and response into input_ids and creates target_ids
        where prompt tokens are masked out with ignore_index.
        """
        input_ids = torch.tensor(prompt_tokens + response_tokens, dtype=torch.long)
        target_ids = torch.tensor([ignore_index] * len(prompt_tokens) + response_tokens, dtype=torch.long)
        return input_ids, target_ids
