import torch
import argparse
from scripts._bootstrap import ROOT_DIR
from llm.tokenizer import BPETokenizer
from llm.config import GPTConfig
from llm.model import GPTLanguageModel
from llm.paths import MODEL_PATH, TOKENIZER_PREFIX
from llm.agent import Agent, Tool

def evaluate_math(expression):
    # Safe evaluate for basic math
    allowed_chars = "0123456789+-*/(). "
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters in expression"
    try:
        return eval(expression)
    except Exception as e:
        return f"Error: {e}"

def search_weather(location):
    # Mock weather tool
    return f"The weather in {location} is currently 72°F and sunny."

def main():
    parser = argparse.ArgumentParser(description='Run agentic chat')
    parser.add_argument('--model_path', type=str, default=str(MODEL_PATH), help='Path to model checkpoint')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PREFIX))
    
    config = GPTConfig()
    config.vocab_size = len(tokenizer.vocab)
    
    print("Loading model...")
    model = GPTLanguageModel(config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    tools = [
        Tool("Calculator", "Evaluates basic math expressions", evaluate_math),
        Tool("Weather", "Gets current weather for a location", search_weather)
    ]
    
    agent = Agent(model, tokenizer, device, tools=tools)
    
    print("\n" + "="*50)
    print("Agent is ready! (Type 'quit' to exit)")
    print("Available tools:", ", ".join([t.name for t in tools]))
    print("="*50)
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input.strip():
                continue
                
            agent.run(user_input)
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
