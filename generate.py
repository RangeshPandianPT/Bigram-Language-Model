import torch
from tokenizer import BPETokenizer
from config import GPTConfig, TrainConfig
from model import GPTLanguageModel

def generate():
    # Load config and arguments
    train_config = TrainConfig()
    gpt_config = GPTConfig()
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load("bpe")
    gpt_config.vocab_size = len(tokenizer.vocab)
    
    # Initialize model
    model = GPTLanguageModel(gpt_config)
    model.load_state_dict(torch.load('model.pth', map_location=train_config.device))
    model.to(train_config.device)
    model.eval()
    
    print("Model loaded.")
    
    # Generate
    print("\nGenerating text:")
    context = torch.zeros((1, 1), dtype=torch.long, device=train_config.device)
    generated_indices = model.generate(context, max_new_tokens=500)[0].tolist()
    print(tokenizer.decode(generated_indices))

if __name__ == "__main__":
    generate()
