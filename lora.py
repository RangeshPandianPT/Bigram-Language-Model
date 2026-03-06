import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.0, **kwargs):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        
        # Base linear layer
        self.linear = nn.Linear(in_features, out_features, **kwargs)
        
        # LoRA adaptation matrices
        if r > 0:
            self.lora_A = nn.Parameter(self.linear.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.linear.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Dropout for LoRA
            self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0. else nn.Identity()
            self.reset_parameters()
            
            # Freeze the pre-trained weight matrix
            self.linear.weight.requires_grad = False
            if self.linear.bias is not None:
                self.linear.bias.requires_grad = False
                
    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # Initialize A similarly to standard weights
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # Initialize B to zero so that the initial effect of LoRA is 0
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        # Base layer path
        result = self.linear(x)
        # LoRA path
        if self.r > 0:
            result += (self.dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        return result

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> nn.Module:
    """
    Freezes all model parameters except those with 'lora_' in their names.
    """
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
            
    if bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALinear) and m.linear.bias is not None:
                m.linear.bias.requires_grad = True
                
    return model

def inject_lora(model: nn.Module, config) -> nn.Module:
    """
    Injects LoRALinear layers into the q_proj and v_proj of the attention blocks
    of an existing GPTLanguageModel.
    """
    if config.lora_rank <= 0:
        return model
        
    for block in model.blocks:
        # q_proj
        orig_q = block.sa.q_proj
        new_q = LoRALinear(orig_q.in_features, orig_q.out_features, 
                           r=config.lora_rank, lora_alpha=config.lora_alpha, 
                           lora_dropout=config.lora_dropout, bias=orig_q.bias is not None)
        new_q.linear.weight.data = orig_q.weight.data.clone()
        if orig_q.bias is not None:
            new_q.linear.bias.data = orig_q.bias.data.clone()
        block.sa.q_proj = new_q
        
        # v_proj
        orig_v = block.sa.v_proj
        new_v = LoRALinear(orig_v.in_features, orig_v.out_features, 
                           r=config.lora_rank, lora_alpha=config.lora_alpha, 
                           lora_dropout=config.lora_dropout, bias=orig_v.bias is not None)
        new_v.linear.weight.data = orig_v.weight.data.clone()
        if orig_v.bias is not None:
            new_v.linear.bias.data = orig_v.bias.data.clone()
        block.sa.v_proj = new_v
        
    return mark_only_lora_as_trainable(model)
