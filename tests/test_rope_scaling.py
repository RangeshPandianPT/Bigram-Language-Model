import torch
from src.llm.model import precompute_freqs_cis

def test_rope_scaling_modes():
    head_dim = 16
    seq_len = 32

    cos_def, sin_def = precompute_freqs_cis(head_dim, seq_len, rope_type="default", rope_scale=1.0)
    cos_lin, sin_lin = precompute_freqs_cis(head_dim, seq_len, rope_type="linear", rope_scale=2.0)
    cos_yarn, sin_yarn = precompute_freqs_cis(head_dim, seq_len, rope_type="yarn", rope_scale=2.0)

    assert cos_def.shape == (seq_len, head_dim // 2)
    assert cos_lin.shape == (seq_len, head_dim // 2)
    assert cos_yarn.shape == (seq_len, head_dim // 2)

    # Linear scaling should alter cos values
    assert not torch.allclose(cos_def, cos_lin)
    assert not torch.allclose(cos_def, cos_yarn)
