import torch
import torch.nn as nn
import torch.nn.functional as F

class Int8Linear(nn.Module):
    """
    INT8 Quantized Linear Layer (Symmetric Per-Channel / Per-Tensor Quantization).
    Weights are quantized to int8: q = clamp(round(w / scale), -128, 127).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, per_channel: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.per_channel = per_channel

        self.register_buffer("weight_int8", torch.zeros((out_features, in_features), dtype=torch.int8))
        self.register_buffer("scale", torch.ones((out_features, 1) if per_channel else (1,), dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

    @classmethod
    def from_float(cls, linear: nn.Linear, per_channel: bool = True):
        int8_linear = cls(linear.in_features, linear.out_features, bias=linear.bias is not None, per_channel=per_channel)
        weight = linear.weight.data.float()

        if per_channel:
            max_val = weight.abs().max(dim=1, keepdim=True)[0]
            scale = torch.clamp(max_val / 127.0, min=1e-8)
        else:
            max_val = weight.abs().max()
            scale = torch.clamp(max_val / 127.0, min=1e-8)

        q_weight = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)

        int8_linear.weight_int8.copy_(q_weight)
        int8_linear.scale.copy_(scale)
        if linear.bias is not None:
            int8_linear.bias.copy_(linear.bias.data.float())

        return int8_linear

    def dequantize_weight(self) -> torch.Tensor:
        return self.weight_int8.float() * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_float = self.dequantize_weight().to(x.dtype)
        return F.linear(x, w_float, self.bias.to(x.dtype) if self.bias is not None else None)


class QuantizedKVCache:
    """
    INT8 Quantized Key-Value Cache buffer for memory-efficient inference generation.
    """
    def __init__(self):
        self.k_int8 = None
        self.k_scale = None
        self.v_int8 = None
        self.v_scale = None

    def update(self, key: torch.Tensor, value: torch.Tensor):
        """
        key, value: (B, T, n_head, head_dim)
        """
        k_max = key.abs().max(dim=-1, keepdim=True)[0]
        v_max = value.abs().max(dim=-1, keepdim=True)[0]

        k_scale = torch.clamp(k_max / 127.0, min=1e-8)
        v_scale = torch.clamp(v_max / 127.0, min=1e-8)

        k_quant = torch.clamp(torch.round(key / k_scale), -128, 127).to(torch.int8)
        v_quant = torch.clamp(torch.round(value / v_scale), -128, 127).to(torch.int8)

        if self.k_int8 is None:
            self.k_int8 = k_quant
            self.k_scale = k_scale
            self.v_int8 = v_quant
            self.v_scale = v_scale
        else:
            self.k_int8 = torch.cat([self.k_int8, k_quant], dim=1)
            self.k_scale = torch.cat([self.k_scale, k_scale], dim=1)
            self.v_int8 = torch.cat([self.v_int8, v_quant], dim=1)
            self.v_scale = torch.cat([self.v_scale, v_scale], dim=1)

    def get_dequantized(self) -> tuple[torch.Tensor, torch.Tensor]:
        k_dequant = self.k_int8.float() * self.k_scale
        v_dequant = self.v_int8.float() * self.v_scale
        return k_dequant, v_dequant


def quantize_model_int8(model: nn.Module) -> nn.Module:
    """
    Replaces all Linear layers in a PyTorch model with Int8Linear layers.
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            setattr(model, name, Int8Linear.from_float(child))
        else:
            quantize_model_int8(child)
    return model
