import torch
from scripts._bootstrap import ROOT_DIR
from llm.config import GPTConfig, TrainConfig
from llm.model import GPTLanguageModel
from llm.tokenizer import BPETokenizer
from llm.paths import TOKENIZER_PREFIX, MODEL_PATH
from scripts.train import train

def train_moe():
    """
    Wrapper around the main training loop that specifically enables
    the Mixture of Experts (MoE) architecture.
    """
    print("🚀 Initializing Mixture of Experts (MoE) Training...")
    
    # 1. Initialize Tokenizer
    tokenizer = BPETokenizer()
    try:
        tokenizer.load(str(TOKENIZER_PREFIX))
    except FileNotFoundError:
        print("❌ Tokenizer not found. Please run prepare_data.py first.")
        return

    # 2. Setup MoE Configuration
    config = GPTConfig(
        vocab_size=len(tokenizer.vocab),
        n_layer=4,           # Depth
        n_head=8,            # Query heads
        n_embd=256,          # Embedding dim
        n_experts=4,         # <--- MoE ENABLED: 4 Experts
        num_experts_per_tok=2 # <--- Route to top 2 experts
    )
    
    train_config = TrainConfig(
        batch_size=16,       # Adjust for memory
        max_iters=5000,
        eval_interval=500,
        learning_rate=3e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"MoE Config: {config.n_experts} experts, {config.num_experts_per_tok} active per token.")
    
    # Initialize model
    model = GPTLanguageModel(config).to(train_config.device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"MoE Model Parameters: {params:.2f}M")
    
    # Run standard training loop with this MoE model
    train(model=model, train_config=train_config)

if __name__ == "__main__":
    train_moe()
