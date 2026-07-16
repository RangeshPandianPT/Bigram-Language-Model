import torch
import argparse
from scripts._bootstrap import ROOT_DIR
from llm.tokenizer import BPETokenizer
from llm.config import GPTConfig
from llm.model import GPTLanguageModel
from llm.paths import MODEL_PATH, TOKENIZER_PREFIX
from llm.agent import Agent, Tool

import urllib.request
import urllib.parse
import json

import io
import sys
import contextlib

def execute_python(code: str):
    """Executes python code and returns stdout. Use this for math and logic."""
    code = code.strip("`")
    if code.startswith("python"): code = code[6:]
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, globals())
        res = output.getvalue()
        return res if res else "Code executed successfully with no output."
    except Exception as e:
        return f"Error: {e}"

def evaluate_math(expression):
    # Safe evaluate for basic math
    allowed_chars = "0123456789+-*/(). "
    if not all(c in allowed_chars for c in expression):
        return "Error: Invalid characters in expression"
    try:
        return eval(expression)
    except Exception as e:
        return f"Error: {e}"

def search_wikipedia(query):
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            if data['query']['search']:
                snippet = data['query']['search'][0]['snippet']
                # clean basic HTML tags from snippet
                snippet = snippet.replace('<span class="searchmatch">', '').replace('</span>', '')
                snippet = snippet.replace('&quot;', '"').replace('&#039;', "'")
                return f"Wikipedia Search Result: {snippet}..."
            else:
                return "No results found on Wikipedia."
    except Exception as e:
        return f"Error searching Wikipedia: {e}"

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
        Tool("Wikipedia", "Searches Wikipedia for a given query", search_wikipedia),
        Tool("PythonREPL", "Executes Python code and returns output", execute_python)
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
