from tokenizer import BPETokenizer

# 1. Load the dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 2. Train the tokenizer
# We'll shoot for a vocab size of 512 + 256 = 768 for this demo (keeps it small/fast)
# GPT-4 uses ~100k, but for Shakespeare 1k-2k is usually plenty.
# Let's try 512 total tokens (256 bytes + 256 merges)
vocab_size = 512

print(f"Training BPE tokenizer on {len(text)} characters...")
print(f"Target vocab size: {vocab_size}")

tokenizer = BPETokenizer()
tokenizer.train(text, vocab_size, verbose=True)

# 3. Save the model
tokenizer.save("bpe")

# 4. Test encoding/decoding
print("\nTesting tokenizer:")
test_str = "Hello, world! This is a test."
encoded = tokenizer.encode(test_str)
decoded = tokenizer.decode(encoded)

print(f"Original: '{test_str}'")
print(f"Encoded: {encoded}")
print(f"Decoded: '{decoded}'")
print(f"Compression ratio: {len(test_str) / len(encoded):.2f}X")

assert test_str == decoded
print("Test passed!")
