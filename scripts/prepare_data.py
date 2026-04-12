import numpy as np

from scripts._bootstrap import ROOT_DIR
from llm.tokenizer import BPETokenizer
from llm.paths import INPUT_TEXT_PATH, TOKENIZER_PREFIX, TRAIN_BIN_PATH, VAL_BIN_PATH, ensure_project_dirs

def prepare_data():
    ensure_project_dirs()

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PREFIX))
    vocab_size = len(tokenizer.vocab)
    print(f"Vocab size: {vocab_size}")

    # Read input text
    with open(INPUT_TEXT_PATH, 'r', encoding='utf-8') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # Helper to encode and save
    def encode_and_save(split_name, text_data, output_path):
        print(f"Processing {split_name} data...")
        ids = tokenizer.encode(text_data)
        print(f"{split_name} has {len(ids)} tokens")
        
        # Save as uint16 (since vocab < 65535)
        # For larger vocabs (e.g. Llama 32k/128k), use uint32
        ids = np.array(ids, dtype=np.uint16)
        ids.tofile(output_path)
        print(f"Saved {output_path}")

    encode_and_save('train', train_data, TRAIN_BIN_PATH)
    encode_and_save('val', val_data, VAL_BIN_PATH)

if __name__ == '__main__':
    prepare_data()
