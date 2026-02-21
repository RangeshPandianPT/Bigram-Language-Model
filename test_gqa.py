"""
test_gqa.py — Verify Grouped Query Attention (GQA) correctness and efficiency.

Tests:
  1. KV memory savings: GQA has fewer KV projection parameters than MHA
  2. Output shape: forward pass produces (B, T, vocab_size) logits
  3. Generation: model.generate() runs without errors with GQA active
"""
import torch
from config import GPTConfig
from model import GPTLanguageModel

DEVICE = "cpu"

# ──────────────────────────────────────────────
# Helper: count KV projection parameters
# ──────────────────────────────────────────────
def count_kv_params(model):
    total = 0
    for block in model.blocks:
        total += block.sa.k_proj.weight.numel()
        total += block.sa.v_proj.weight.numel()
    return total

# ──────────────────────────────────────────────
# Test 1: GQA has fewer KV params than MHA
# ──────────────────────────────────────────────
def test_gqa_kv_param_reduction():
    base = dict(block_size=64, vocab_size=256, n_layer=2, n_head=4, n_embd=64)

    # MHA: n_kv_head == n_head
    mha_cfg = GPTConfig(**base, n_kv_head=4)
    mha_model = GPTLanguageModel(mha_cfg).to(DEVICE)
    mha_kv_params = count_kv_params(mha_model)

    # GQA: n_kv_head < n_head
    gqa_cfg = GPTConfig(**base, n_kv_head=2)
    gqa_model = GPTLanguageModel(gqa_cfg).to(DEVICE)
    gqa_kv_params = count_kv_params(gqa_model)

    reduction = (1 - gqa_kv_params / mha_kv_params) * 100
    print(f"  MHA KV params : {mha_kv_params:,}")
    print(f"  GQA KV params : {gqa_kv_params:,}  ({reduction:.1f}% reduction)")
    assert gqa_kv_params < mha_kv_params, "GQA should have fewer KV params than MHA"
    assert abs(reduction - 50.0) < 1e-3, f"Expected ~50% reduction, got {reduction:.1f}%"
    print("  ✅ PASSED: KV parameter reduction is correct")

# ──────────────────────────────────────────────
# Test 2: Forward pass output shape is correct
# ──────────────────────────────────────────────
def test_gqa_forward_shape():
    cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=4, n_embd=64, n_kv_head=2)
    model = GPTLanguageModel(cfg).to(DEVICE)
    model.eval()

    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T)).to(DEVICE)
    targets = torch.randint(0, cfg.vocab_size, (B, T)).to(DEVICE)

    with torch.no_grad():
        logits, loss, past_kv = model(idx, targets)

    assert logits.shape == (B * T, cfg.vocab_size), \
        f"Expected ({B*T}, {cfg.vocab_size}), got {logits.shape}"
    assert loss is not None and loss.item() > 0, "Loss should be positive"
    assert len(past_kv) == cfg.n_layer, "Should have one KV tuple per layer"

    # Verify KV cache shapes use n_kv_head
    head_dim = cfg.n_embd // cfg.n_head
    k, v = past_kv[0]
    assert k.shape == (B, T, cfg.n_kv_head, head_dim), \
        f"K cache shape mismatch: {k.shape}"
    print(f"  Logits shape  : {logits.shape}")
    print(f"  KV cache K[0] : {k.shape}  (n_kv_head={cfg.n_kv_head})")
    print("  ✅ PASSED: Output shapes are correct")

# ──────────────────────────────────────────────
# Test 3: Generation runs end-to-end
# ──────────────────────────────────────────────
def test_gqa_generation():
    cfg = GPTConfig(block_size=64, vocab_size=256, n_layer=2, n_head=4, n_embd=64, n_kv_head=2)
    model = GPTLanguageModel(cfg).to(DEVICE)
    model.eval()

    torch.manual_seed(42)
    prompt = torch.randint(0, cfg.vocab_size, (1, 5)).to(DEVICE)
    output = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=10)

    assert output.shape[1] == 5 + 20, \
        f"Expected {5+20} tokens, got {output.shape[1]}"
    print(f"  Generated sequence length: {output.shape[1]} tokens")
    print("  ✅ PASSED: Generation with GQA works correctly")

# ──────────────────────────────────────────────
# Run all tests
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Grouped Query Attention (GQA) Tests ===\n")

    print("[Test 1] KV Parameter Reduction:")
    test_gqa_kv_param_reduction()

    print("\n[Test 2] Forward Pass Output Shape:")
    test_gqa_forward_shape()

    print("\n[Test 3] End-to-End Generation:")
    test_gqa_generation()

    print("\n🎉 All GQA tests passed!")
