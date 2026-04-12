import onnxruntime as ort
import numpy as np
import torch
import os
from llm.config import GPTConfig
from llm.model import GPTLanguageModel

def verify_onnx():
    print("Loading PyTorch model...")
    checkpoint = torch.load('model.pth', map_location='cpu')
    config = GPTConfig()
    
    if 'token_embedding_table.weight' in checkpoint:
        vocab_size, n_embd = checkpoint['token_embedding_table.weight'].shape
        config.vocab_size = vocab_size
        config.n_embd = n_embd
    
    n_layer = sum(1 for k in checkpoint if k.endswith('.sa.c_proj.weight'))
    if n_layer > 0:
        config.n_layer = n_layer
        
    if config.n_embd == 128:
        config.n_head = 4; config.n_kv_head = 2; config.block_size = 64
    elif config.n_embd == 256:
        config.n_head = 8; config.n_kv_head = 4; config.block_size = 128
        
    model = GPTLanguageModel(config)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    print("Loading ONNX model...")
    ort_session = ort.InferenceSession("model.onnx")

    print("Generating random input...")
    # Use block_size to match the sequence length used during ONNX export tracing
    dummy_input = torch.randint(0, config.vocab_size, (1, config.block_size))
    
    print("Running PyTorch...")
    with torch.no_grad():
        pt_logits, _, _ = model(dummy_input)
    
    print("Running ONNX...")
    ort_inputs = {"input_ids": dummy_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_logits = ort_outs[0]

    print("Comparing outputs...")
    # Adjust tolerance if needed
    np.testing.assert_allclose(pt_logits.numpy(), onnx_logits, rtol=1e-03, atol=1e-04)
    print("Success! ONNX model outputs match PyTorch within numerical tolerance.")

if __name__ == "__main__":
    verify_onnx()
