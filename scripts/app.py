"""
app.py — Interactive web demo for the GPT Language Model
Run: python app.py
"""
import torch
import gradio as gr
import os

from scripts._bootstrap import ROOT_DIR
from llm.config import GPTConfig, SamplingConfig
from llm.model import GPTLanguageModel
from llm.tokenizer import BPETokenizer
from llm.paths import MODEL_BEST_PATH, MODEL_PATH, TOKENIZER_PREFIX

# ─────────────────────────────────────────────
# Load model and tokenizer on startup
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BPETokenizer()
tokenizer.load(str(TOKENIZER_PREFIX))

config = GPTConfig()
config.vocab_size = len(tokenizer.vocab)

model = GPTLanguageModel(config).to(DEVICE)

# Try best checkpoint first, fall back to model.pth
CKPT = str(MODEL_BEST_PATH) if os.path.exists(str(MODEL_BEST_PATH)) else str(MODEL_PATH)
if os.path.exists(CKPT):
    state = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    print(f"✅ Loaded weights from {CKPT}")
else:
    print("⚠️  No checkpoint found — using random weights (run train.py first!)")

model.eval()
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model: {config.n_layer}L / {config.n_embd}E / {config.n_head}H | {params:.2f}M params | device={DEVICE}")

# ─────────────────────────────────────────────
# Generation function
# ─────────────────────────────────────────────
@torch.no_grad()
def generate_text(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
):
    if not prompt.strip():
        return "⚠️ Please enter a prompt to start generation."

    try:
        tokens = tokenizer.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

        # Crop if prompt exceeds block_size
        if idx.shape[1] > config.block_size:
            idx = idx[:, -config.block_size:]

        output_ids = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else 0,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        full_text = tokenizer.decode(output_ids[0].tolist())
        # Highlight the generated part
        generated_only = tokenizer.decode(output_ids[0, len(tokens):].tolist())
        return f"{prompt}[GENERATED:]{generated_only}"

    except Exception as e:
        return f"❌ Error during generation: {e}"

# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────
CSS = """
#header { text-align: center; margin-bottom: 10px; }
#output-box textarea { font-family: 'Georgia', serif; font-size: 15px; line-height: 1.7; }
"""

with gr.Blocks(
    title="GPT Language Model Demo",
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"),
    css=CSS,
) as demo:

    gr.Markdown(
        """
        # 🧠 GPT Language Model — Interactive Demo
        A **LLaMA-style** GPT trained on Shakespeare — featuring RMSNorm, RoPE, SwiGLU, GQA & KV Cache.
        """,
        elem_id="header",
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Generation Settings")

            temperature = gr.Slider(
                0.1, 2.0, value=0.8, step=0.05,
                label="🌡️ Temperature",
                info="Higher = more creative / random. Lower = more focused.",
            )
            top_k = gr.Slider(
                0, 100, value=40, step=1,
                label="🔝 Top-K",
                info="Keep only top K tokens each step. 0 = disabled.",
            )
            top_p = gr.Slider(
                0.1, 1.0, value=0.9, step=0.01,
                label="🎯 Top-P (Nucleus)",
                info="Cumulative probability cutoff. 1.0 = disabled.",
            )
            rep_penalty = gr.Slider(
                1.0, 2.0, value=1.2, step=0.05,
                label="🔁 Repetition Penalty",
                info="Penalizes repeated tokens. 1.0 = no penalty.",
            )
            max_tokens = gr.Slider(
                10, 300, value=100, step=10,
                label="📏 Max New Tokens",
            )

        with gr.Column(scale=2):
            gr.Markdown("### ✍️ Prompt & Output")

            prompt_box = gr.Textbox(
                label="Prompt",
                placeholder="Enter your starting text here…",
                lines=4,
                value="ROMEO:\nWhat light through yonder window breaks",
            )

            output_box = gr.Textbox(
                label="Generated Text  (prompt + [GENERATED:] continuation)",
                lines=12,
                interactive=False,
                elem_id="output-box",
            )

            with gr.Row():
                generate_btn = gr.Button("🚀 Generate", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear", scale=1)

    gr.Markdown(
        f"""
        ---
        **Model specs:** `{config.n_layer}` layers · `{config.n_embd}` embed dim · \
`{config.n_head}` query heads · `{config.n_kv_head}` KV heads (GQA) · \
`{config.block_size}` context · **{params:.2f}M params** · device: `{DEVICE}`
        """,
    )

    # Wire up buttons
    generate_btn.click(
        fn=generate_text,
        inputs=[prompt_box, max_tokens, temperature, top_k, top_p, rep_penalty],
        outputs=output_box,
    )
    clear_btn.click(fn=lambda: ("", ""), outputs=[prompt_box, output_box])

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
