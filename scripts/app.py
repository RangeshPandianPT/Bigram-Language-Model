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
from llm.rag import DocumentLoader, TextChunker, VectorStore
from scripts.speculative_decode import speculative_decode
from llm.agent import Agent, Tool
from scripts.agent_chat import evaluate_math, search_wikipedia, execute_python
import time

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
# Initialize RAG VectorStore
# ─────────────────────────────────────────────
vector_store = None
doc_dir = os.path.join(ROOT_DIR, 'data', 'documents')
try:
    if os.path.exists(doc_dir):
        loader = DocumentLoader()
        docs = loader.load_directory(doc_dir)
        if docs:
            chunker = TextChunker()
            chunks = []
            for doc in docs:
                chunks.extend(chunker.chunk_document(doc))
            vector_store = VectorStore()
            vector_store.ingest(chunks)
            print(f"✅ RAG VectorStore initialized with {len(chunks)} chunks.")
except Exception as e:
    print(f"⚠️ RAG VectorStore failed to initialize: {e}")

# ─────────────────────────────────────────────
# Initialize Draft Model for Speculative Decoding
# ─────────────────────────────────────────────
draft_config = GPTConfig(n_layer=2, n_embd=64, n_head=2, vocab_size=config.vocab_size)
draft_model = GPTLanguageModel(draft_config).to(DEVICE)
draft_model.eval()
print(f"✅ Draft Model initialized for Speculative Decoding.")

# ─────────────────────────────────────────────
# Generation functions
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

        for next_token in model.generate_stream(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else 0,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        ):
            idx = torch.cat((idx, next_token), dim=1)
            # Highlight the generated part
            generated_only = tokenizer.decode(idx[0, len(tokens):].tolist())
            yield f"{prompt}[GENERATED:]{generated_only}"

    except Exception as e:
        yield f"❌ Error during generation: {e}"

@torch.no_grad()
def generate_rag(prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    if not prompt.strip(): return "⚠️ Please enter a prompt."
    if not vector_store: return "⚠️ VectorStore not initialized. Please add .txt files to data/documents."
    
    try:
        results = vector_store.search(prompt, top_k=3)
        context_str = "\n".join([res['content'] for res in results]) if results else ""
        
        full_prompt = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context_str}\n\nQuestion: {prompt}\nAnswer:"
        ) if context_str else f"Question: {prompt}\nAnswer:"

        tokens = tokenizer.encode(full_prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
        
        for next_token in model.generate_stream(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p):
            idx = torch.cat((idx, next_token), dim=1)
            generated_only = tokenizer.decode(idx[0, len(tokens):].tolist())
            yield f"**Context Retrieved:**\n{context_str}\n\n**Answer:**\n{generated_only.strip()}"
    except Exception as e:
        yield f"❌ Error: {e}"

@torch.no_grad()
def generate_speculative(prompt: str, max_new_tokens: int):
    if not prompt.strip(): return "⚠️ Please enter a prompt."
    try:
        tokens = tokenizer.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=DEVICE)
        
        t0 = time.time()
        std_out = model.generate(idx, max_new_tokens=max_new_tokens)[0]
        std_time = time.time() - t0
        
        t0 = time.time()
        spec_out = speculative_decode(model, draft_model, idx, tokenizer, max_new_tokens=max_new_tokens)[0]
        spec_time = time.time() - t0
        
        std_text = tokenizer.decode(std_out[len(tokens):].tolist())
        spec_text = tokenizer.decode(spec_out[len(tokens):].tolist())
        
        return (f"**Standard Generation ({std_time:.2f}s):**\n{std_text}\n\n"
                f"**Speculative Decoding ({spec_time:.2f}s):**\n{spec_text}\n\n"
                f"*Speedup: {std_time/spec_time:.2f}x*")
    except Exception as e:
        return f"❌ Error: {e}"

# ─────────────────────────────────────────────
# Agent Chat function
# ─────────────────────────────────────────────
agent_tools = [
    Tool("Calculator", "Evaluates basic math expressions", evaluate_math),
    Tool("Wikipedia", "Searches Wikipedia for a given query", search_wikipedia),
    Tool("PythonREPL", "Executes Python code", execute_python)
]
agent = Agent(model, tokenizer, DEVICE, tools=agent_tools)

def agent_chat_fn(prompt: str, history):
    if not prompt.strip(): return history
    try:
        response = agent.run(prompt)
        history.append((prompt, response))
        return "", history
    except Exception as e:
        history.append((prompt, f"❌ Error: {e}"))
        return "", history

# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────
CSS = """
#header { text-align: center; margin-bottom: 10px; }
#output-box textarea { font-family: 'Georgia', serif; font-size: 15px; line-height: 1.7; }
"""

with gr.Blocks(title="GPT Language Model Demo", theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple"), css=CSS) as demo:
    gr.Markdown("# 🧠 GPT Language Model — Interactive Demo\nA **LLaMA-style** GPT trained on Shakespeare — featuring RMSNorm, RoPE, SwiGLU, GQA, RAG, & Speculative Decoding.", elem_id="header")

    with gr.Tabs():
        with gr.TabItem("✍️ Text Generation"):
            with gr.Row():
                with gr.Column(scale=1):
                    temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="🌡️ Temperature")
                    top_k = gr.Slider(0, 100, value=40, step=1, label="🔝 Top-K")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.01, label="🎯 Top-P")
                    rep_penalty = gr.Slider(1.0, 2.0, value=1.2, step=0.05, label="🔁 Repetition Penalty")
                    max_tokens = gr.Slider(10, 300, value=100, step=10, label="📏 Max Tokens")
                with gr.Column(scale=2):
                    prompt_box = gr.Textbox(label="Prompt", value="ROMEO:\nWhat light through yonder window breaks")
                    output_box = gr.Textbox(label="Generated Text", interactive=False, lines=10)
                    gen_btn = gr.Button("🚀 Generate", variant="primary")
            gen_btn.click(generate_text, [prompt_box, max_tokens, temperature, top_k, top_p, rep_penalty], output_box)

        with gr.TabItem("📚 Chat with Docs (RAG)"):
            with gr.Row():
                with gr.Column(scale=1):
                    rag_max_tokens = gr.Slider(10, 300, value=150, step=10, label="📏 Max Tokens")
                    rag_temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.05, label="🌡️ Temperature")
                    rag_top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.01, label="🎯 Top-P")
                with gr.Column(scale=2):
                    rag_prompt = gr.Textbox(label="Question", placeholder="Ask something about your documents...")
                    rag_output = gr.Textbox(label="Answer & Context", interactive=False, lines=10)
                    rag_btn = gr.Button("🔍 Retrieve & Generate", variant="primary")
            rag_btn.click(generate_rag, [rag_prompt, rag_max_tokens, rag_temperature, rag_top_p], rag_output)

        with gr.TabItem("⚡ Speculative Decoding"):
            with gr.Row():
                with gr.Column(scale=1):
                    spec_max_tokens = gr.Slider(10, 300, value=50, step=10, label="📏 Max Tokens")
                with gr.Column(scale=2):
                    spec_prompt = gr.Textbox(label="Prompt", value="The future of artificial intelligence is")
                    spec_output = gr.Textbox(label="Comparison", interactive=False, lines=10)
                    spec_btn = gr.Button("🏎️ Test Speed", variant="primary")
            spec_btn.click(generate_speculative, [spec_prompt, spec_max_tokens], spec_output)

        with gr.TabItem("🤖 Agent Chat"):
            chatbot = gr.Chatbot(label="Agent (Uses Tools: Python, Wiki, Math)")
            with gr.Row():
                agent_prompt = gr.Textbox(show_label=False, placeholder="Ask the agent to calculate, search, or code...", scale=4)
                agent_btn = gr.Button("Send", variant="primary", scale=1)
            agent_btn.click(agent_chat_fn, [agent_prompt, chatbot], [agent_prompt, chatbot])

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
