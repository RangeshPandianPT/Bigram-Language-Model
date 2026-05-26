import torch
import torch.nn.functional as F
from tqdm import tqdm
from scripts._bootstrap import ROOT_DIR
from llm.tokenizer import BPETokenizer
from llm.config import GPTConfig, TrainConfig
from llm.model import GPTLanguageModel
from llm.paths import MODEL_PATH, TOKENIZER_PREFIX
from scripts.eval.tasks import TASKS

def evaluate(task_name, model, tokenizer, device, num_samples=100):
    task_cls = TASKS[task_name]
    task = task_cls()
    dataset = task.get_dataset(split='validation' if task_name != 'mmlu' else 'test')
    
    correct = 0
    total = 0
    
    for i, example in enumerate(tqdm(dataset, desc=f"Evaluating {task_name}")):
        if i >= num_samples:
            break
            
        context, choices, label = task.format_example(example)
        if label == -1 or not choices:
            continue
            
        context_ids = tokenizer.encode(context)
        
        choice_losses = []
        for choice in choices:
            choice_ids = tokenizer.encode(" " + choice)
            
            # evaluate loss of choice given context
            full_ids = context_ids + choice_ids
            input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
            
            with torch.no_grad():
                logits, _, _ = model(input_tensor)
                
            # we only care about the loss on the choice tokens
            shift_logits = logits[0, len(context_ids)-1:-1, :]
            shift_labels = torch.tensor(choice_ids, dtype=torch.long, device=device)
            
            loss = F.cross_entropy(shift_logits, shift_labels, reduction='sum')
            choice_losses.append(loss.item())
            
        pred_label = choice_losses.index(min(choice_losses))
        if pred_label == label:
            correct += 1
        total += 1
        
    accuracy = correct / total if total > 0 else 0.0
    print(f"{task_name} Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    return accuracy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=str(MODEL_PATH))
    parser.add_argument('--tasks', type=str, default='piqa', help='comma separated tasks')
    parser.add_argument('--num_samples', type=int, default=100)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PREFIX))
    
    config = GPTConfig()
    config.vocab_size = len(tokenizer.vocab)
    model = GPTLanguageModel(config)
    
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    tasks = args.tasks.split(',')
    for t in tasks:
        if t in TASKS:
            evaluate(t, model, tokenizer, device, num_samples=args.num_samples)
        else:
            print(f"Task {t} not found in TASKS.")

if __name__ == "__main__":
    main()
