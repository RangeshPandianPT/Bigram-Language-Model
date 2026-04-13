# Llama-from-Scratch

A pedagogical implementation of a modern Large Language Model (LLM) architecture, evolving from a simple GPT-2 style baseline to a Llama 2/3 style model.

## 🚀 Project Goal
To understand modern LLM internals by building them from scratch in PyTorch. This project bridges the gap between basic transformer tutorials (like Andrej Karpathy's `nanogpt`) and production-grade architectures like Llama 3.

## ✨ Key Features (Implemented)

### 🏗️ Architecture ("Llama-fication" & "Mixtral-style")
- **RMSNorm** (Root Mean Square Normalization): Replaces LayerNorm for better stability and performance.
- **RoPE** (Rotary Positional Embeddings): Replaces absolute positional embeddings for better generalization to longer sequences.
- **SwiGLU** (SiLU Gated Linear Unit): A more powerful activation function in the FeedForward network.
- **Grouped Query Attention (GQA)**: Optimizes inference memory by sharing Key/Value heads across multiple Query heads (Llama 2/3 style).
- **Mixture of Experts (MoE)**: Sparse routing mechanism for drastically increasing parameter count with constant compute (Mixtral 8x-style).

### 🛠️ Engineering
- **Modular Codebase**: Separated into `config.py`, `model.py`, `train.py`, `tokenizer.py`.
- **Mixed Precision Training**: Implemented via `torch.amp` for faster training on modern GPUs.
- **KV Cache**: Efficient O(N) generation inference.
- **Efficient Data Loading**: Uses `numpy.memmap` for handling datasets larger than RAM.

### 🧠 Tokenization
- Uses a custom BPE (Byte Pair Encoding) tokenizer trained on the dataset.

### 🚀 Fine-Tuning & Deployment
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning script (`train_lora.py`) freezing base weights and injecting rank-decomposed matrices.
- **ONNX Export**: Script (`export_onnx.py`) to export the PyTorch model for efficient cross-platform and edge device inference.
## 📂 File Structure

| File | Purpose |
| --- | --- |
| `model.py` | Core Transformer: RMSNorm, RoPE, GQA, SwiGLU, KV Cache |
| `train.py` | Training loop with AMP, LR scheduling, gradient clipping |
| `config.py` | GPTConfig, TrainConfig, SamplingConfig dataclasses |
| `generate.py` | CLI inference with advanced sampling |
| `app.py` | 🌐 Gradio web demo |
| `tokenizer.py` | BPE tokenizer wrapper |
| `prepare_data.py` | Tokenizes raw text → `train.bin` / `val.bin` |
| `test_gqa.py` | GQA unit tests (KV param reduction + output shape) |

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python prepare_data.py
```

### 3. Train
```bash
python train.py
```

### 4. Generate Text (CLI)
```bash
python generate.py
```

## 🌐 Web Demo

Launch an interactive web interface to generate text with live parameter controls:

```bash
python app.py
```

Then open [http://localhost:7860](http://localhost:7860) in your browser.

**Demo controls:**

| Control | Description |
| --- | --- |
| 🌡️ Temperature | Creativity/randomness (0.1 = focused, 2.0 = wild) |
| 🔝 Top-K | Keep only the top K most likely tokens |
| 🎯 Top-P | Nucleus sampling — smallest set with cumulative prob ≥ P |
| 🔁 Repetition Penalty | Discourages repeating tokens |
| 📏 Max New Tokens | How many tokens to generate |

## 🧪 Testing

```bash
python test_gqa.py        # GQA correctness: KV param reduction + output shape
python test_kv_cache.py   # KV Cache: verify cached == uncached outputs + speedup
python test_new_features.py  # Mixed precision, sampling strategies, LR scheduler
```

## 📜 License
MIT
