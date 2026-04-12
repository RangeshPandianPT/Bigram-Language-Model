import os

def check_leakage():
    output = []
    with open('input.txt', 'r', encoding='utf-8') as f:
        data = f.read()

    n = len(data)
    split_idx = int(n * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    output.append(f"Total characters: {n}")
    output.append(f"Train size: {len(train_data)}")
    output.append(f"Val size: {len(val_data)}")

    # Check 1: Is the validation set EXACTLY present in the train set?
    if val_data in train_data:
        output.append("CRITICAL: The entire validation set is present in the training set!")
        # Find where it occurs
        idx = train_data.find(val_data)
        output.append(f"Validation set found in training set at index {idx}")
    else:
        output.append("Validation set is NOT exactly present in the training set.")

    # Check 2: Check for overlap of meaningful chunks (e.g., 100 chars)
    # We'll take the first 1000 chars of validation and see if they exist in train
    chunk_size = 1000
    if len(val_data) > chunk_size:
        val_start = val_data[:chunk_size]
        if val_start in train_data:
            output.append(f"WARNING: The first {chunk_size} characters of validation set appear in the training set.")
            idx = train_data.find(val_start)
            output.append(f"Found at index {idx}")
            output.append(f"Snippet: {val_start[:50]}...")
        else:
            output.append(f"The first {chunk_size} characters of validation set do NOT appear in the training set.")

    # Check 3: Check if the file is a repetition of itself
    # e.g. does the second half equal the first half?
    mid = n // 2
    first_half = data[:mid]
    second_half = data[mid:2*mid]
    
    if first_half == second_half:
         output.append("CRITICAL: The file appears to be two identical concatenated halves.")
    
    # Check for smaller repetitions
    # Let's take a sample from the beginning and see if it repeats
    sample = data[:500]
    count = data.count(sample)
    if count > 1:
         output.append(f"WARNING: The first 500 characters appear {count} times in the dataset.")
         # Find indices
         import re
         indices = [m.start() for m in re.finditer(re.escape(sample), data)]
         output.append(f"Indices: {indices}")
    
    with open('leakage_report.txt', 'w') as f:
        f.write('\n'.join(output))

if __name__ == "__main__":
    check_leakage()
