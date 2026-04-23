import os
import json
import urllib.request
import numpy as np

from scripts._bootstrap import ROOT_DIR
from llm.tokenizer import BPETokenizer
from llm.paths import (
    RAW_DATA_DIR,
    TOKENIZER_PREFIX,
    CHAT_TRAIN_BIN_PATH,
    CHAT_VAL_BIN_PATH,
    ensure_project_dirs
)

ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
LOCAL_ALPACA_PATH = RAW_DATA_DIR / "alpaca_data.json"

def download_alpaca():
    if not LOCAL_ALPACA_PATH.exists():
        print(f"Downloading Alpaca dataset from {ALPACA_URL}...")
        try:
            urllib.request.urlretrieve(ALPACA_URL, LOCAL_ALPACA_PATH)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            return None
    return LOCAL_ALPACA_PATH

def format_instruction(entry):
    instruction = entry.get('instruction', '')
    input_text = entry.get('input', '')
    output_text = entry.get('output', '')
    
    if input_text:
        user_text = f"{instruction}\n\n{input_text}"
    else:
        user_text = instruction
        
    # We use simple string markers because BPETokenizer might not have special tokens registered
    return f"User: {user_text}\nAssistant: {output_text}\n<|endoftext|>\n"

def prepare_chat_data():
    ensure_project_dirs()

    # 1. Download
    dataset_path = download_alpaca()
    if not dataset_path:
        return

    # 2. Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Found {len(data)} examples. Subsetting to 500 for fast pedagogical tokenization.")
    data = data[:500]
    
    # Optional: subset for pedagogical purposes to speed up tokenization if needed
    # data = data[:5000]
    
    # 3. Format as chat
    formatted_data = []
    for entry in data:
        formatted_data.append(format_instruction(entry))
        
    full_text = "".join(formatted_data)
    print(f"Total formatted characters: {len(full_text)}")

    # 4. Tokenize
    print("Loading tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PREFIX))
    vocab_size = len(tokenizer.vocab)
    print(f"Vocab size: {vocab_size}")

    def encode_and_save(split_name, text_data_list, output_path):
        print(f"Tokenizing {split_name} data in chunks...")
        all_ids = []
        for i, text_chunk in enumerate(text_data_list):
            if i % 5000 == 0:
                print(f"  Processed {i}/{len(text_data_list)} items...")
            all_ids.extend(tokenizer.encode(text_chunk))
        
        print(f"{split_name} has {len(all_ids)} tokens")
        ids = np.array(all_ids, dtype=np.uint16)
        ids.tofile(output_path)
        print(f"Saved {output_path}")

    # For splitting, we just use the list of formatted data instead of a single massive string
    n = len(formatted_data)
    train_data_list = formatted_data[:int(n * 0.9)]
    val_data_list = formatted_data[int(n * 0.9):]

    encode_and_save('train', train_data_list, CHAT_TRAIN_BIN_PATH)
    encode_and_save('val', val_data_list, CHAT_VAL_BIN_PATH)

if __name__ == "__main__":
    prepare_chat_data()
