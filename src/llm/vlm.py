import torch
import torch.nn as nn
from llm.config import GPTConfig
from llm.model import GPTLanguageModel

class VisionEncoder(nn.Module):
    """ Dummy vision encoder (e.g. CLIP substitute) """
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        # In reality, this would be a ResNet or ViT
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        # Assume 224x224 image -> 56x56 -> 3136 * 64 -> map to hidden_dim sequence
        # For simplicity, let's just make it a linear layer that projects the whole image to a single embedding sequence
        self.proj = nn.Linear(64 * 56 * 56, hidden_dim * 16) # 16 image patches/tokens
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: (B, 3, 224, 224)
        B = x.size(0)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.proj(x)
        x = x.view(B, 16, self.hidden_dim) # (B, num_image_tokens, vision_dim)
        return x

class VLM(nn.Module):
    """ Vision-Language Model combining VisionEncoder and GPTLanguageModel """
    def __init__(self, config: GPTConfig, vision_dim: int = 512):
        super().__init__()
        self.config = config
        
        # Language Model
        self.llm = GPTLanguageModel(config)
        
        # Vision Encoder
        self.vision_encoder = VisionEncoder(hidden_dim=vision_dim)
        
        # Projection layer: Vision Dim -> LLM Embedding Dim
        self.mm_projector = nn.Linear(vision_dim, config.n_embd)
        
    def forward(self, input_ids, images=None, targets=None):
        B, T = input_ids.shape
        
        # 1. Get text embeddings
        text_embeds = self.llm.token_embedding_table(input_ids) # (B, T, C)
        
        # 2. Add image embeddings if provided
        if images is not None:
            # Get vision features
            vision_features = self.vision_encoder(images) # (B, num_img_toks, vision_dim)
            
            # Project to LLM dimension
            img_embeds = self.mm_projector(vision_features) # (B, num_img_toks, n_embd)
            
            # Prepend image embeddings to text embeddings
            # In a real LLaVA model, we would insert them at specific <image> token positions
            inputs_embeds = torch.cat([img_embeds, text_embeds], dim=1)
            
            # If targets are provided, pad targets to account for image tokens
            if targets is not None:
                img_targets = torch.full((B, img_embeds.size(1)), -100, dtype=torch.long, device=targets.device)
                targets = torch.cat([img_targets, targets], dim=1)
        else:
            inputs_embeds = text_embeds
            
        # 3. Forward through the rest of the LLM using embeddings
        # (We need to modify the LLM to accept inputs_embeds directly, or just copy the logic here)
        # For simplicity, we'll bypass the embedding layer of the LLM and pass it to blocks
        
        x = inputs_embeds
        seq_len = x.size(1)
        
        # Calculate RoPE
        freqs_cos = self.llm.freqs_cos[:seq_len]
        freqs_sin = self.llm.freqs_sin[:seq_len]
        
        for block in self.llm.blocks:
            x, _ = block(x, freqs_cos, freqs_sin)
            
        x = self.llm.ln_f(x)
        logits = self.llm.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T_out, C = logits.shape
            logits = logits.view(B*T_out, C)
            targets = targets.view(B*T_out)
            loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-100)
            
        return logits, loss
