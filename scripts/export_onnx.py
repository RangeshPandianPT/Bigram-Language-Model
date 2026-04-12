import torch
import os
from scripts._bootstrap import ROOT_DIR
from llm.config import GPTConfig
from llm.model import GPTLanguageModel
from llm.tokenizer import BPETokenizer
from llm.paths import MODEL_PATH, ONNX_MODEL_PATH, TOKENIZER_PREFIX, ensure_project_dirs

def export_to_onnx(model_path=str(MODEL_PATH), output_path=str(ONNX_MODEL_PATH)):
    ensure_project_dirs()

    print("Loading config and tokenizer...")
    print("Loading tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PREFIX))
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}'. Please ensure the model is trained.")
        return
        
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Infer config from checkpoint
    config = GPTConfig()
    if 'token_embedding_table.weight' in checkpoint:
        vocab_size, n_embd = checkpoint['token_embedding_table.weight'].shape
        config.vocab_size = vocab_size
        config.n_embd = n_embd
    
    n_layer = sum(1 for k in checkpoint if k.endswith('.sa.c_proj.weight'))
    if n_layer > 0:
        config.n_layer = n_layer
        
    if 'blocks.0.sa.c_proj.weight' in checkpoint:
        # shape: (n_embd, n_head * head_dim) but c_proj is (n_embd, n_head * head_dim)
        # where head_dim = n_embd // n_head.
        # Actually it's just (n_embd, n_embd)
        pass # Can't trivially infer n_head, we'll try config.n_head or default (old was 4, new is 8)
        
    if config.n_embd == 128:
        config.n_head = 4
        config.n_kv_head = 2
        config.block_size = 64
    elif config.n_embd == 256:
        config.n_head = 8
        config.n_kv_head = 4
        config.block_size = 128
        
    print(f"Initializing model with vocab_size={config.vocab_size}, n_embd={config.n_embd}, n_layer={config.n_layer}...")
    model = GPTLanguageModel(config)
    
    # strict=False in case of freqs_cos/sin buffer mismatch
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    print("Preparing dummy input...")
    # Dummy input: (batch_size=1, seq_len=config.block_size)
    dummy_input = torch.randint(0, config.vocab_size, (1, config.block_size))
    
    print(f"Exporting to {output_path}...")
    
    # We create a simple wrapper module to export just the logits prediction
    # since ONNX export prefers fixed number of outputs without nested structures (loss is None here)
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, idx):
            logits, _, _ = self.model(idx)
            return logits

    wrapper = ONNXWrapper(model)
    
    torch.onnx.export(
        wrapper, 
        (dummy_input,),
        output_path, 
        export_params=True, 
        opset_version=14, 
        do_constant_folding=True, 
        input_names=['input_ids'], 
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    print("Export successful!")

if __name__ == "__main__":
    export_to_onnx()
