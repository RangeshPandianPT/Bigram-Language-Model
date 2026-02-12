# Project Roadmap: Next Level LLM

Your current implementation is a solid "Vanilla GPT" (similar to GPT-2). To take this project to the next level (modern Open LLaMA style), we can improve three key areas: **Architecture**, **Engineering**, and **Inference**.

## Level 1: Architectural Modernization ("Llama-fication")
Modern LLMs have moved away from the original GPT-2 architecture in several ways to improve stability and performance.

### 1. RMSNorm (Root Mean Square Normalization)
**Why:** Replaces LayerNorm. It's simpler, faster, and more numerically stable. Used in Llama, Gopher, Chinchilla.
**Task:** Replace `nn.LayerNorm` with a custom `RMSNorm` module.

### 2. Rotary Positional Embeddings (RoPE)
**Why:** Replaces absolute learned positional embeddings. RoPE encodes relative positions better, allowing the model to generalize to sequence lengths longer than it was trained on.
**Task:** Implement rotary embedding application to Q and K vectors in `SelfAttention`.

### 3. SwiGLU Activation
**Why:** Replaces ReLU or GeLU. It consistently yields better performance for the same compute budget.
**Task:** Update the `FeedForward` validation to use the SwiGLU variant (requires a different parameter structure).

### 4. Grouped Query Attention (GQA)
**Why:** Reduces memory bandwidth usage during inference. It's a middle ground between Multi-Head (MHA) and Multi-Query (MQA).
**Task:** Modify `MultiHeadAttention` to share keys and values across multiple heads.

---

## Level 2: Scalable Engineering
Your current `bigram.py` does everything in one file. Real-world projects need modularity and efficiency.

### 1. Modularity & Refactoring
**Why:** `bigram.py` contains the model definition, training loop, and data loading.
**Task:**
- `model.py`: The GPT class and modules.
- `train.py`: The training loop and saving logic.
- `config.py`: Hyperparameters.
- `generate.py`: Standalone inference script.

### 2. Efficient Data Loading (`numpy.memmap`)
**Why:** Currently, you load `input.txt` into RAM (`encoded_data = tokenizer.encode(text)`). This crashes with large datasets (e.g., OpenWebText, 10GB+).
**Task:** Pre-tokenize the dataset into a binary file (`.bin`) and use `numpy.memmap` to load slices from disk without eating RAM.

### 3. Mixed Precision Training
**Why:** Uses `float16` or `bfloat16` for faster training and less memory usage.
**Task:** Integrate `torch.amp` (Automatic Mixed Precision) and `scaler` into the training loop.

---

## Level 3: Inference Optimization
Making the model generate text faster and smarter.

### 1. KV Optimization (Key-Value Cache)
**Why:** In generation, we re-compute Attention for previous tokens every step. KV Cache saves these calculations, making generation O(N) instead of O(N^2).
**Task:** Modify the `forward` pass to accept and update a `past_key_values` cache.

### 2. Sampling Strategies
**Why:** Standard `multinomial` sampling can be chaotic.
**Task:** Implement **Temperature scaling**, **Top-k**, and **Top-p (Nucleus)** sampling for higher quality generation.
