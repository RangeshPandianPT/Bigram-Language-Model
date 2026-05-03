import argparse
import torch
import json
from tqdm import tqdm

from scripts._bootstrap import ROOT_DIR
from llm.tokenizer import BPETokenizer
from llm.config import GPTConfig, TrainConfig
from llm.model import GPTLanguageModel
from llm.paths import MODEL_PATH, TOKENIZER_PREFIX
from scripts.eval.tasks import TASKS
from scripts.eval.metrics import get_choice_log_likelihood

def evaluate(args):
    train_config = TrainConfig()
    gpt_config = GPTConfig()
    device = train_config.device
    
    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PREFIX))
    gpt_config.vocab_size = len(tokenizer.vocab)
    
    model = GPTLanguageModel(gpt_config)
    model_path = args.model_path if args.model_path else str(MODEL_PATH)
    
    if not args.untrained:
        try:
            state_dict = torch.load(model_path, map_location=device)
            try:
                model.load_state_dict(state_dict)
                print(f"Loaded model from {model_path}")
            except RuntimeError as e:
                print("Architecture mismatch with saved checkpoint. The model architecture has changed (e.g. RoPE, GQA added) since this checkpoint was saved.")
                print("Falling back to randomly initialized weights for testing. Please retrain the model to get valid results.")
        except FileNotFoundError:
            print(f"Model file not found at {model_path}. Proceeding with untrained model for testing.")
    else:
        print("Using randomly initialized (untrained) model as requested.")
        
    model.to(device)
    model.eval()
    
    results = {}
    
    tasks_to_run = args.tasks.split(",")
    for task_name in tasks_to_run:
        task_name = task_name.strip()
        if task_name not in TASKS:
            print(f"Task {task_name} not found. Available tasks: {list(TASKS.keys())}")
            continue
            
        print(f"Evaluating {task_name}...")
        task = TASKS[task_name]()
        try:
            dataset = task.get_dataset(split='validation')
        except Exception as e:
            print(f"Failed to load dataset for {task_name}: {e}")
            continue
            
        if args.limit and args.limit > 0:
            dataset = dataset.select(range(min(args.limit, len(dataset))))
            
        correct = 0
        total = len(dataset)
        
        for example in tqdm(dataset, desc=task_name):
            context, choices, label = task.format_example(example)
            
            best_choice_idx = -1
            best_log_likelihood = -float('inf')
            
            for idx, choice in enumerate(choices):
                # Add a leading space to the choice as is standard in continuation tasks
                choice_text = " " + choice if not choice.startswith(" ") else choice
                ll = get_choice_log_likelihood(model, tokenizer, context, choice_text, device)
                if ll > best_log_likelihood:
                    best_log_likelihood = ll
                    best_choice_idx = idx
            
            if best_choice_idx == label:
                correct += 1
                
        accuracy = correct / total if total > 0 else 0
        print(f"[{task_name}] Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
        results[task_name] = {"accuracy": accuracy, "correct": correct, "total": total}
        
    output_file = ROOT_DIR / "reports" / "eval_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the model on standard benchmarks')
    parser.add_argument("--model_path", type=str, default=str(MODEL_PATH), help="Path to model checkpoint")
    parser.add_argument("--tasks", type=str, default="hellaswag,piqa", help="Comma-separated list of tasks")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of examples per task (0 for all)")
    parser.add_argument("--untrained", action="store_true", help="Run with an untrained, randomly initialized model")
    args = parser.parse_args()
    evaluate(args)
