import json
import os
from datasets import load_dataset
from scripts._bootstrap import ROOT_DIR

def main():
    print("Loading Anthropic/hh-rlhf dataset from Hugging Face...")
    try:
        # Load a tiny subset just for demonstration to keep it fast
        dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    data_dir = os.path.join(ROOT_DIR, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "preference_data.json")

    formatted_data = []
    
    for row in dataset:
        # The HH-RLHF dataset has 'chosen' and 'rejected' columns which contain the full conversation.
        # We need to extract the final assistant response.
        chosen = row['chosen']
        rejected = row['rejected']
        
        # A simple hack to split prompt from the response.
        # Anthropic data format usually ends with "\n\nAssistant: [response]"
        split_token = "\n\nAssistant: "
        
        if split_token in chosen and split_token in rejected:
            prompt_chosen_split = chosen.rpartition(split_token)
            prompt = prompt_chosen_split[0] + prompt_chosen_split[1]
            chosen_response = prompt_chosen_split[2]
            
            prompt_rejected_split = rejected.rpartition(split_token)
            rejected_response = prompt_rejected_split[2]
            
            formatted_data.append({
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response
            })
            
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully processed and saved {len(formatted_data)} preference pairs to {out_path}")
    print("You can now run 'python scripts/train_dpo.py' to train on real data!")

if __name__ == "__main__":
    main()
