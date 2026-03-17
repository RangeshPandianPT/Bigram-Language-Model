"""
Automated evaluation scorecard for model variants.

Usage examples:
  python generate_report.py
  python generate_report.py --compare base lora onnx --max-new-tokens 120
  python generate_report.py --temperature 0.8 --top-k 50 --top-p 0.9
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

from config import GPTConfig, TrainConfig
from lora import inject_lora
from model import GPTLanguageModel
from tokenizer import BPETokenizer

try:
    import onnxruntime as ort
except ImportError:
    ort = None


DEFAULT_PROMPTS = [
    "To be, or not to be",
    "The king said",
    "In fair Verona",
]


@dataclass
class EvalResult:
    name: str
    perplexity: float
    latency_ms_per_token: float
    repetition_rate: float
    distinct_1: float
    distinct_2: float


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def infer_config_from_checkpoint(checkpoint: Dict[str, torch.Tensor]) -> GPTConfig:
    config = GPTConfig()

    if "token_embedding_table.weight" in checkpoint:
        vocab_size, n_embd = checkpoint["token_embedding_table.weight"].shape
        config.vocab_size = vocab_size
        config.n_embd = n_embd

    n_layer = sum(1 for k in checkpoint if k.endswith(".sa.c_proj.weight"))
    if n_layer > 0:
        config.n_layer = n_layer

    # Preserve existing project assumptions for known model sizes.
    if config.n_embd == 128:
        config.n_head = 4
        config.n_kv_head = 2
        config.block_size = 64
    elif config.n_embd == 256:
        config.n_head = 8
        config.n_kv_head = 4
        config.block_size = 128

    return config


def load_tokenizer(prefix: str) -> BPETokenizer:
    tokenizer = BPETokenizer()
    tokenizer.load(prefix)
    return tokenizer


def load_base_model(model_path: str, device: str, vocab_size: int) -> Tuple[GPTLanguageModel, GPTConfig]:
    checkpoint = torch.load(model_path, map_location="cpu")
    config = infer_config_from_checkpoint(checkpoint)
    config.vocab_size = vocab_size

    model = GPTLanguageModel(config)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    return model, config


def infer_lora_rank(lora_state: Dict[str, torch.Tensor]) -> int:
    for key, value in lora_state.items():
        if key.endswith("lora_A"):
            return int(value.shape[0])
    return 8


def load_lora_model(base_model_path: str, lora_path: str, device: str, vocab_size: int) -> Tuple[GPTLanguageModel, GPTConfig]:
    checkpoint = torch.load(base_model_path, map_location="cpu")
    lora_state = torch.load(lora_path, map_location="cpu")

    config = infer_config_from_checkpoint(checkpoint)
    config.vocab_size = vocab_size
    config.lora_rank = infer_lora_rank(lora_state)
    config.lora_alpha = 32
    config.lora_dropout = 0.0

    model = GPTLanguageModel(config)
    model.load_state_dict(checkpoint, strict=False)
    model = inject_lora(model, config)
    model.load_state_dict(lora_state, strict=False)
    model.to(device)
    model.eval()
    return model, config


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def sample_next_token_numpy(
    logits: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    generated: List[int],
) -> int:
    logits = logits.astype(np.float64)

    if repetition_penalty != 1.0 and generated:
        for token_id in set(generated):
            logits[token_id] /= repetition_penalty

    logits = logits / max(temperature, 1e-6)

    if top_k > 0:
        kth = np.partition(logits, -top_k)[-top_k]
        logits[logits < kth] = -np.inf

    if top_p < 1.0:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        sorted_probs = softmax_np(sorted_logits)
        cumulative = np.cumsum(sorted_probs)
        remove = cumulative > top_p
        remove[1:] = remove[:-1]
        remove[0] = False
        logits[sorted_indices[remove]] = -np.inf

    probs = softmax_np(logits)
    next_token = int(np.random.choice(len(probs), p=probs))
    return next_token


@torch.no_grad()
def sample_next_token_torch(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    generated: List[int],
) -> int:
    logits = logits.clone()

    if repetition_penalty != 1.0 and generated:
        for token_id in set(generated):
            logits[token_id] /= repetition_penalty

    logits = logits / max(temperature, 1e-6)

    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[-1]] = -float("inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float("inf")

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_with_torch(
    model: GPTLanguageModel,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    device: str,
) -> List[int]:
    if prompt_ids:
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated = idx[0].tolist()
    past_key_values = None

    for _ in range(max_new_tokens):
        if past_key_values is not None:
            idx_cond = idx[:, -1:]
        else:
            idx_cond = idx[:, -model.config.block_size :]

        logits, _, past_key_values = model(idx_cond, past_key_values=past_key_values)
        next_token = sample_next_token_torch(
            logits[0, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated=generated,
        )

        next_token_t = torch.tensor([[next_token]], dtype=torch.long, device=device)
        idx = torch.cat((idx, next_token_t), dim=1)
        generated.append(next_token)

    return generated


def generate_with_onnx(
    session: "ort.InferenceSession",
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    context_len: int,
) -> List[int]:
    generated = list(prompt_ids) if prompt_ids else [0]

    for _ in range(max_new_tokens):
        # Keep a fixed ONNX context length to match export-time assumptions.
        window = generated[-context_len:]
        if len(window) < context_len:
            window = ([0] * (context_len - len(window))) + window

        input_ids = np.array([window], dtype=np.int64)
        logits = session.run(None, {"input_ids": input_ids})[0]
        next_token = sample_next_token_numpy(
            logits=logits[0, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated=generated,
        )
        generated.append(next_token)

    return generated


def distinct_n(tokens: List[int], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / max(len(ngrams), 1)


def repetition_rate(tokens: List[int]) -> float:
    if not tokens:
        return 0.0
    repeats = len(tokens) - len(set(tokens))
    return repeats / len(tokens)


@torch.no_grad()
def perplexity_torch(
    model: GPTLanguageModel,
    val_data: np.memmap,
    block_size: int,
    batch_size: int,
    steps: int,
    device: str,
) -> float:
    losses = []

    for _ in range(steps):
        ix = np.random.randint(0, len(val_data) - block_size - 1, size=(batch_size,))
        x = np.stack([val_data[i : i + block_size].astype(np.int64) for i in ix])
        y = np.stack([val_data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix])

        x_t = torch.from_numpy(x).to(device)
        y_t = torch.from_numpy(y).to(device)

        _, loss, _ = model(x_t, y_t)
        losses.append(float(loss.item()))

    return float(math.exp(np.mean(losses)))


def logsumexp_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m


def perplexity_onnx(
    session: "ort.InferenceSession",
    val_data: np.memmap,
    block_size: int,
    batch_size: int,
    steps: int,
) -> float:
    if len(val_data) <= block_size + 1:
        raise ValueError(
            f"val.bin is too small for ONNX context length {block_size}. "
            f"Need at least {block_size + 2} tokens, found {len(val_data)}."
        )

    losses = []

    for _ in range(steps):
        ix = np.random.randint(0, len(val_data) - block_size - 1, size=(batch_size,))
        x = np.stack([val_data[i : i + block_size].astype(np.int64) for i in ix])
        y = np.stack([val_data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix])

        logits = session.run(None, {"input_ids": x})[0]
        lse = logsumexp_np(logits, axis=-1)

        b_idx = np.arange(batch_size)[:, None]
        t_idx = np.arange(block_size)[None, :]
        target_logits = logits[b_idx, t_idx, y]

        nll = -(target_logits - lse.squeeze(-1))
        losses.append(float(np.mean(nll)))

    return float(math.exp(np.mean(losses)))


def resolve_onnx_context_len(session: "ort.InferenceSession", requested: int) -> int:
    candidates = [requested, 128, 64, 32]
    tried = []

    for context_len in dict.fromkeys(candidates):
        if context_len <= 0:
            continue
        try:
            probe = np.zeros((1, context_len), dtype=np.int64)
            session.run(None, {"input_ids": probe})
            return context_len
        except Exception as exc:  # pragma: no cover - runtime dependent
            tried.append(f"{context_len}: {exc}")

    details = " | ".join(tried)
    raise RuntimeError(f"Could not resolve ONNX context length. Attempts: {details}")


def evaluate_model(
    name: str,
    generate_fn: Callable[[List[int]], List[int]],
    perplexity_fn: Callable[[], float],
    tokenizer: BPETokenizer,
    max_new_tokens: int,
) -> EvalResult:
    ppl = perplexity_fn()

    latencies = []
    rep_rates = []
    d1s = []
    d2s = []

    for prompt in DEFAULT_PROMPTS:
        prompt_ids = tokenizer.encode(prompt)

        start = time.perf_counter()
        out_ids = generate_fn(prompt_ids)
        duration = time.perf_counter() - start

        new_tokens = out_ids[len(prompt_ids) :]
        if not new_tokens:
            new_tokens = out_ids[-max_new_tokens:]

        latencies.append((duration * 1000.0) / max(len(new_tokens), 1))
        rep_rates.append(repetition_rate(new_tokens))
        d1s.append(distinct_n(new_tokens, 1))
        d2s.append(distinct_n(new_tokens, 2))

    return EvalResult(
        name=name,
        perplexity=float(np.mean([ppl])),
        latency_ms_per_token=float(np.mean(latencies)),
        repetition_rate=float(np.mean(rep_rates)),
        distinct_1=float(np.mean(d1s)),
        distinct_2=float(np.mean(d2s)),
    )


def render_markdown(results: List[EvalResult], output_path: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Evaluation Scorecard",
        "",
        f"Generated: {now}",
        "",
        "| Model | Perplexity (lower better) | Latency ms/token (lower better) | Repetition Rate (lower better) | Distinct-1 (higher better) | Distinct-2 (higher better) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for r in results:
        lines.append(
            f"| {r.name} | {r.perplexity:.4f} | {r.latency_ms_per_token:.3f} | {r.repetition_rate:.4f} | {r.distinct_1:.4f} | {r.distinct_2:.4f} |"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_summary(results: List[EvalResult], output_path: str) -> None:
    print("\n" + "=" * 98)
    print("EVALUATION SCORECARD")
    print("=" * 98)
    print(
        f"{'Model':<10} {'Perplexity':>12} {'ms/token':>12} {'Repetition':>12} {'Distinct-1':>12} {'Distinct-2':>12}"
    )
    print("-" * 98)

    for r in results:
        print(
            f"{r.name:<10} {r.perplexity:>12.4f} {r.latency_ms_per_token:>12.3f} {r.repetition_rate:>12.4f} {r.distinct_1:>12.4f} {r.distinct_2:>12.4f}"
        )

    print("=" * 98)
    print(f"Saved markdown report to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base/LoRA/ONNX model variants.")
    parser.add_argument("--compare", nargs="+", default=["base", "lora", "onnx"], choices=["base", "lora", "onnx"]) 
    parser.add_argument("--model-path", type=str, default="model.pth")
    parser.add_argument("--lora-path", type=str, default="lora_weights.pth")
    parser.add_argument("--onnx-path", type=str, default="model.onnx")
    parser.add_argument("--tokenizer-prefix", type=str, default="bpe")
    parser.add_argument("--val-bin", type=str, default="val.bin")
    parser.add_argument("--output", type=str, default="evaluation_report.md")

    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--ppl-steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--onnx-context-len", type=int, default=64)

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    if not os.path.exists(args.val_bin):
        raise FileNotFoundError(f"Validation binary not found: {args.val_bin}")

    tokenizer = load_tokenizer(args.tokenizer_prefix)
    val_data = np.memmap(args.val_bin, dtype=np.uint16, mode="r")

    train_config = TrainConfig()
    device = train_config.device

    results: List[EvalResult] = []

    if "base" in args.compare:
        if not os.path.exists(args.model_path):
            print(f"[WARN] Skipping base: missing {args.model_path}")
        else:
            base_model, base_config = load_base_model(args.model_path, device, len(tokenizer.vocab))

            base_result = evaluate_model(
                name="base",
                generate_fn=lambda prompt_ids: generate_with_torch(
                    model=base_model,
                    prompt_ids=prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    device=device,
                ),
                perplexity_fn=lambda: perplexity_torch(
                    model=base_model,
                    val_data=val_data,
                    block_size=base_config.block_size,
                    batch_size=args.batch_size,
                    steps=args.ppl_steps,
                    device=device,
                ),
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
            )
            results.append(base_result)

    if "lora" in args.compare:
        if not os.path.exists(args.model_path) or not os.path.exists(args.lora_path):
            print(f"[WARN] Skipping lora: missing {args.model_path} or {args.lora_path}")
        else:
            lora_model, lora_config = load_lora_model(args.model_path, args.lora_path, device, len(tokenizer.vocab))

            lora_result = evaluate_model(
                name="lora",
                generate_fn=lambda prompt_ids: generate_with_torch(
                    model=lora_model,
                    prompt_ids=prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    device=device,
                ),
                perplexity_fn=lambda: perplexity_torch(
                    model=lora_model,
                    val_data=val_data,
                    block_size=lora_config.block_size,
                    batch_size=args.batch_size,
                    steps=args.ppl_steps,
                    device=device,
                ),
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
            )
            results.append(lora_result)

    if "onnx" in args.compare:
        if ort is None:
            print("[WARN] Skipping onnx: onnxruntime is not installed")
        elif not os.path.exists(args.onnx_path):
            print(f"[WARN] Skipping onnx: missing {args.onnx_path}")
        else:
            onnx_session = ort.InferenceSession(args.onnx_path)
            onnx_context_len = resolve_onnx_context_len(onnx_session, args.onnx_context_len)
            if onnx_context_len != args.onnx_context_len:
                print(
                    f"[WARN] Requested ONNX context {args.onnx_context_len} is incompatible; "
                    f"using {onnx_context_len}"
                )
            print(f"[INFO] ONNX context length: {onnx_context_len}")

            onnx_result = evaluate_model(
                name="onnx",
                generate_fn=lambda prompt_ids: generate_with_onnx(
                    session=onnx_session,
                    prompt_ids=prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    context_len=onnx_context_len,
                ),
                perplexity_fn=lambda: perplexity_onnx(
                    session=onnx_session,
                    val_data=val_data,
                    block_size=onnx_context_len,
                    batch_size=args.batch_size,
                    steps=args.ppl_steps,
                ),
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
            )
            results.append(onnx_result)

    if not results:
        raise RuntimeError("No model variants were evaluated. Check file paths and dependencies.")

    render_markdown(results, args.output)
    print_summary(results, args.output)


if __name__ == "__main__":
    main()
