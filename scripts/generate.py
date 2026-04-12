import torch
import argparse

from scripts._bootstrap import ROOT_DIR
from llm.tokenizer import BPETokenizer
from llm.config import GPTConfig, TrainConfig
from llm.model import GPTLanguageModel
from llm.paths import MODEL_PATH, TOKENIZER_PREFIX

def generate(args):
    # Load config
    train_config = TrainConfig()
    gpt_config = GPTConfig()
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PREFIX))
    gpt_config.vocab_size = len(tokenizer.vocab)
    
    # Initialize model
    model = GPTLanguageModel(gpt_config)
    
    # Load model weights
    model_path = args.model_path if args.model_path else str(MODEL_PATH)
    model.load_state_dict(torch.load(model_path, map_location=train_config.device))
    model.to(train_config.device)
    model.eval()
    
    if args.quantize:
        print("Applying dynamic quantization to int8...")
        if train_config.device == 'cuda':
            print("Warning: Dynamic Quantization is optimized for CPU inference. Moving model to CPU.")
            model.to('cpu')
            train_config.device = 'cpu'
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    
    print(f"Model loaded from {model_path}")
    print(f"Sampling config: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, repetition_penalty={args.repetition_penalty}")
    
    # Generate
    if args.prompt:
        # Encode the prompt
        prompt_ids = tokenizer.encode(args.prompt)
        context = torch.tensor([prompt_ids], dtype=torch.long, device=train_config.device)
        print(f"\nPrompt: {args.prompt}")
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=train_config.device)
        print("\nGenerating from scratch:")
    
    print("-" * 80)
    generated_indices = model.generate(
        context, 
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty
    )[0].tolist()
    print(tokenizer.decode(generated_indices))
    print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text from trained model')
    parser.add_argument('--model_path', type=str, default=str(MODEL_PATH), help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='', help='Prompt text to start generation')
    parser.add_argument('--max_tokens', type=int, default=500, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=0, help='Top-k sampling (0 = disabled)')
    parser.add_argument('--top_p', type=float, default=1.0, help='Nucleus sampling (1.0 = disabled)')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='Repetition penalty (1.0 = no penalty)')
    parser.add_argument('--quantize', action='store_true', help='Apply dynamic quantization (int8) for faster CPU inference')
    
    args = parser.parse_args()
    generate(args)

