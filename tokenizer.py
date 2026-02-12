import regex as re
import json

# Reference: https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py
# Reference: https://github.com/openai/gpt-2/blob/master/src/encoder.py

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class BPETokenizer:
    def __init__(self):
        self.merges = {} # (int, int) -> int
        self.vocab = {} # int -> bytes
        self.special_tokens = {} # str -> int

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes) # list of integers in range 0..255

        # iteratively merge the most common pair
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]})")

        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # no more merges possible
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.merges
        """
        # save the vocab (for debugging mostly, and to know what bytes each token maps to)
        # We can just save the validation mapping (token_id -> bytes string)
        # But JSON doesn't support integer keys nicely, or bytes values.
        # We'll stick to a simple custom format or just pickle/json carefully.
        # Let's save merges as strict JSON list of lists/tuples, and skip vocab for now (can be reconstructed)
        # Actually, let's keep it simple.
        
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write merges: each line is "idx1 idx2" (space separated) implies merge to next available idx
            for (p0, p1), idx in self.merges.items():
                f.write(f"{p0} {p1}\n")
        print(f"saved to {model_file}")

    def load(self, file_prefix):
        model_file = file_prefix + ".model"
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        
        with open(model_file, 'r') as f:
            for i, line in enumerate(f):
                p0, p1 = map(int, line.split())
                idx = 256 + i
                self.merges[(p0, p1)] = idx
                self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
