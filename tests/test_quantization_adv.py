import torch
import torch.nn as nn
from src.llm.quantization import Int8Linear, QuantizedKVCache, quantize_model_int8

def test_int8_linear_quantization():
    linear = nn.Linear(32, 64)
    int8_lin = Int8Linear.from_float(linear)

    x = torch.randn(2, 32)
    orig_out = linear(x)
    quant_out = int8_lin(x)

    # Check shape and reasonable quantization error bound
    assert quant_out.shape == (2, 64)
    error = torch.mean(torch.abs(orig_out - quant_out))
    assert error < 0.5


def test_quantized_kv_cache():
    cache = QuantizedKVCache()
    k1 = torch.randn(2, 4, 4, 16)
    v1 = torch.randn(2, 4, 4, 16)

    cache.update(k1, v1)
    k_deq, v_deq = cache.get_dequantized()

    assert k_deq.shape == k1.shape
    assert v_deq.shape == v1.shape
