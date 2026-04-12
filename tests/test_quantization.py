import torch
import time
import os
from llm.config import GPTConfig
from llm.model import GPTLanguageModel

def measure_size_and_time(model, context, n_tokens=50):
    # Measure size using state_dict
    torch.save(model.state_dict(), 'temp_model.pth')
    size_mb = os.path.getsize('temp_model.pth') / (1024 * 1024)
    if os.path.exists('temp_model.pth'):
        os.remove('temp_model.pth')
    
    # Warmup
    _ = model.generate(context, max_new_tokens=2, temperature=1.0)
    
    # Measure time
    start_time = time.time()
    _ = model.generate(context, max_new_tokens=n_tokens, temperature=1.0)
    end_time = time.time()
    
    return size_mb, end_time - start_time

def main():
    print("Testing Dynamic Quantization...")
    config = GPTConfig()
    model = GPTLanguageModel(config)
    
    # If a trained model exists, load it
    if os.path.exists('model.pth'):
        try:
            model.load_state_dict(torch.load('model.pth', map_location='cpu'))
            print("Loaded trained model 'model.pth'.")
        except Exception as e:
            print(f"Could not load 'model.pth', using randomly initialized model: {e}")
            model = GPTLanguageModel(config)
    else:
        print("Using randomly initialized model for testing.")
        
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long)
    
    print("\n--- Standard Model (FP32) ---")
    size_fp32, time_fp32 = measure_size_and_time(model, context)
    print(f"Size: {size_fp32:.2f} MB")
    print(f"Time for 50 tokens: {time_fp32:.2f} seconds")
    
    print("\n--- Quantized Model (INT8) ---")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    size_int8, time_int8 = measure_size_and_time(quantized_model, context)
    print(f"Size: {size_int8:.2f} MB")
    print(f"Time for 50 tokens: {time_int8:.2f} seconds")
    
    size_reduction = (1 - size_int8 / size_fp32) * 100
    speedup = time_fp32 / time_int8
    
    print("\n--- Results ---")
    print(f"Model Size Reduction: {size_reduction:.2f}%")
    print(f"Inference Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()
