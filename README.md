# Llama-from-Scratch

A pedagogical implementation of a modern Large Language Model (LLM) architecture, evolving from a simple GPT-2 style baseline to a Llama 2/3 style model.

## 🚀 Project Goal
To understand modern LLM internals by building them from scratch in PyTorch. This project bridges the gap between basic transformer tutorials (like Andrej Karpathy's `nanogpt`) and production-grade architectures like Llama 3.

## ✨ Key Features (Implemented)

### 🏗️ Architecture ("Llama-fication")
- **RMSNorm** (Root Mean Square Normalization): Replaces LayerNorm for better stability and performance.
- **RoPE** (Rotary Positional Embeddings): Replaces absolute positional embeddings for better generalization to longer sequences.
- **SwiGLU** (SiLU Gated Linear Unit): A more powerful activation function in the FeedForward network.
- **Grouped Query Attention (GQA)**: Optimizes inference memory by sharing Key/Value heads across multiple Query heads (Llama 2/3 style).

### 🛠️ Engineering
- **Modular Codebase**: Separated into `config.py`, `model.py`, `train.py`, `tokenizer.py`.
- **Mixed Precision Training**: Implemented via `torch.amp` for faster training on modern GPUs.
- **KV Cache**: Efficient O(N) generation inference.
- **Efficient Data Loading**: Uses `numpy.memmap` for handling datasets larger than RAM.

### 🧠 Tokenization
- Uses a custom BPE (Byte Pair Encoding) tokenizer trained on the dataset.

## 📂 File Structure
- `model.py`: The core Transformer architecture (RMSNorm, RoPE, CausalSelfAttention with GQA, SwiGLU).
- `train.py`: Training loop with learning rate scheduling, gradient clipping, and AMP.
- `config.py`: Configuration classes for Model, Training, and Sampling.
- `generate.py`: Inference script with advanced sampling (Top-k, Top-p, Temperature).
- `prepare_data.py`: Tokenizes raw text into binary `.bin` files for training.
- `investigate_leakage.py`: Utility to check for data leakage between train/val splits.

## ⚡ Quick Start

### 1. Prepare Data
Put your text data in `input.txt` and run:
```bash
python prepare_data.py
```
This will create `train.bin` and `val.bin`.

### 2. Train
```bash
python train.py
```
Adjust parameters in `config.py` or directly in `train.py`.

### 3. Generate Text
```bash
python generate.py
```

## 🧪 Testing
Run the feature verification suite:
```bash
python test_new_features.py
```
This tests:
- GQA forward pass dimensions
- Mixed precision training
- Learning rate scheduler
- Sampling strategies

## 📜 License
MIT
