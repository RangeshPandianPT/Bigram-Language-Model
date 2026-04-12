import torch

from scripts._bootstrap import ROOT_DIR
from llm.tokenizer import BPETokenizer
from llm.config import GPTConfig, TrainConfig
from llm.model import GPTLanguageModel
from llm.paths import MODEL_PATH, TOKENIZER_PREFIX

def compare_sampling_strategies():
    """Compare different sampling strategies side-by-side"""
    
    # Load config
    train_config = TrainConfig()
    gpt_config = GPTConfig()
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PREFIX))
    gpt_config.vocab_size = len(tokenizer.vocab)
    
    # Initialize model
    model = GPTLanguageModel(gpt_config)
    model.load_state_dict(torch.load(str(MODEL_PATH), map_location=train_config.device))
    model.to(train_config.device)
    model.eval()
    
    print("=" * 100)
    print("SAMPLING STRATEGY COMPARISON")
    print("=" * 100)
    
    # Define different sampling configurations
    configs = [
        {"name": "Default (temperature=1.0)", "temp": 1.0, "top_k": 0, "top_p": 1.0, "rep_pen": 1.0},
        {"name": "Low Temperature (0.7)", "temp": 0.7, "top_k": 0, "top_p": 1.0, "rep_pen": 1.0},
        {"name": "Top-k (k=50)", "temp": 1.0, "top_k": 50, "top_p": 1.0, "rep_pen": 1.0},
        {"name": "Nucleus (p=0.9)", "temp": 1.0, "top_k": 0, "top_p": 0.9, "rep_pen": 1.0},
        {"name": "Combined (temp=0.8, top_k=50, top_p=0.9)", "temp": 0.8, "top_k": 50, "top_p": 0.9, "rep_pen": 1.0},
        {"name": "With Repetition Penalty", "temp": 0.8, "top_k": 50, "top_p": 0.9, "rep_pen": 1.2},
    ]
    
    max_tokens = 200
    
    for config in configs:
        print(f"\n{'=' * 100}")
        print(f"Strategy: {config['name']}")
        print(f"Parameters: temp={config['temp']}, top_k={config['top_k']}, top_p={config['top_p']}, rep_penalty={config['rep_pen']}")
        print("-" * 100)
        
        context = torch.zeros((1, 1), dtype=torch.long, device=train_config.device)
        generated_indices = model.generate(
            context,
            max_new_tokens=max_tokens,
            temperature=config['temp'],
            top_k=config['top_k'],
            top_p=config['top_p'],
            repetition_penalty=config['rep_pen']
        )[0].tolist()
        
        print(tokenizer.decode(generated_indices))
        print("-" * 100)
    
    print("\n" + "=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    compare_sampling_strategies()
