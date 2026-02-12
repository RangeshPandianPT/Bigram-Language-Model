import os
import numpy as np
from tokenizer import BPETokenizer

def prepare_data():
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load("bpe")
    vocab_size = len(tokenizer.vocab)
    print(f"Vocab size: {vocab_size}")

    # Read input text
    with open('input.txt', 'r', encoding='utf-8') as f:
        data = f.read()
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # Helper to encode and save
    def encode_and_save(split_name, text_data):
        print(f"Processing {split_name} data...")
        ids = tokenizer.encode(text_data)
        print(f"{split_name} has {len(ids)} tokens")
        
        # Save as uint16 (since vocab < 65535)
        # For larger vocabs (e.g. Llama 32k/128k), use uint32
        ids = np.array(ids, dtype=np.uint16)
        filename = f'{split_name}.bin'
        ids.tofile(filename)
        print(f"Saved {filename}")

    encode_and_save('train', train_data)
    encode_and_save('val', val_data)

if __name__ == '__main__':
    prepare_data()
