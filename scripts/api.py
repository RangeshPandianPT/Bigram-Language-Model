from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import uvicorn
from contextlib import asynccontextmanager

from scripts._bootstrap import ROOT_DIR
from llm.tokenizer import BPETokenizer
from llm.config import GPTConfig, TrainConfig
from llm.model import GPTLanguageModel
from llm.paths import MODEL_BEST_PATH, MODEL_PATH, TOKENIZER_PREFIX
from llm.rag import DocumentLoader, TextChunker, VectorStore
from scripts.speculative_decode import speculative_decode
from llm.agent import Agent, Tool
from scripts.agent_chat import evaluate_math, search_wikipedia, execute_python

# Global variables to hold our model and tokenizer
model = None
draft_model = None
tokenizer = None
device = None
vector_store = None


def _load_model_weights(model_instance: GPTLanguageModel, target_device: str) -> str | None:
    checkpoint_path = MODEL_BEST_PATH if MODEL_BEST_PATH.exists() else MODEL_PATH
    if not checkpoint_path.exists():
        return None

    model_instance.load_state_dict(torch.load(checkpoint_path, map_location=target_device))
    return str(checkpoint_path)


def _build_vector_store() -> VectorStore | None:
    doc_dir = ROOT_DIR / 'data' / 'documents'
    if not doc_dir.exists():
        return None

    loader = DocumentLoader()
    docs = loader.load_directory(str(doc_dir))
    if not docs:
        return None

    chunker = TextChunker()
    chunks = []
    for doc in docs:
        chunks.extend(chunker.chunk_document(doc))

    store = VectorStore()
    store.ingest(chunks)
    return store

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and tokenizer on startup
    global model, tokenizer, device
    print("Loading model and tokenizer...")
    try:
        train_config = TrainConfig()
        gpt_config = GPTConfig()
        device = train_config.device

        tokenizer = BPETokenizer()
        tokenizer.load(str(TOKENIZER_PREFIX))
        gpt_config.vocab_size = len(tokenizer.vocab)

        model = GPTLanguageModel(gpt_config)
        
        # Load the trained weights
        model_path = _load_model_weights(model, device)
        if model_path is None:
            raise FileNotFoundError("No trained model checkpoint found.")
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {model_path}!")
        
        # Initialize Vector Store
        global vector_store
        try:
            vector_store = _build_vector_store()
            if vector_store is not None:
                print("VectorStore initialized from data/documents.")
            else:
                print("VectorStore not initialized - no documents found.")
        except ImportError as e:
            print(f"VectorStore error: {e}")
            
        # Load draft model for speculative decoding
        global draft_model
        try:
            draft_config = GPTConfig(n_layer=2, n_embd=64, n_head=2)
            draft_config.vocab_size = gpt_config.vocab_size
            draft_model = GPTLanguageModel(draft_config)
            
            draft_path = MODEL_PATH.parent / "draft_model.pt"
            if draft_path.exists():
                draft_model.load_state_dict(torch.load(draft_path, map_location=device))
                print("Draft model loaded from artifacts.")
            else:
                print("Draft model initialized (using random weights for demo).")
                
            draft_model.to(device)
            draft_model.eval()
        except Exception as e:
            print(f"Error loading draft model: {e}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        # In a real app we might want to exit here, but for development let's allow it to start up
    
    yield
    # Cleanup on shutdown (if any)
    print("Shutting down...")

app = FastAPI(lifespan=lifespan, title="Bigram LLM API", description="API for generating text using our trained custom LLM")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 200
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    use_speculative_decoding: bool = False
    use_rag: bool = False

class IngestRequest(BaseModel):
    doc_dir: str

class AgentRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "Bigram LLM API is running. Send POST requests to /generate."}


@app.get("/health")
def read_health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "draft_model_loaded": draft_model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "vector_store_loaded": vector_store is not None,
        "device": str(device) if device is not None else None,
        "checkpoint": str(MODEL_BEST_PATH if MODEL_BEST_PATH.exists() else MODEL_PATH),
    }


@app.get("/model-info")
def read_model_info():
    return {
        "checkpoint": str(MODEL_BEST_PATH if MODEL_BEST_PATH.exists() else MODEL_PATH),
        "device": str(device) if device is not None else None,
        "has_vector_store": vector_store is not None,
        "supports_speculative_decoding": draft_model is not None,
    }

@app.post("/ingest")
def ingest_documents(request: IngestRequest):
    global vector_store
    if vector_store is None:
        raise HTTPException(status_code=500, detail="VectorStore not initialized.")
    
    loader = DocumentLoader()
    docs = loader.load_directory(request.doc_dir)
    if not docs:
        raise HTTPException(status_code=400, detail=f"No documents found in {request.doc_dir}")
        
    chunker = TextChunker()
    chunks = []
    for doc in docs:
        chunks.extend(chunker.chunk_document(doc))
        
    vector_store.ingest(chunks)
    return {"status": "success", "chunks_ingested": len(chunks)}

@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet or failed to load.")
    
    try:
        final_prompt = request.prompt
        
        # 1. RAG Context Retrieval
        if request.use_rag and vector_store is not None and request.prompt:
            results = vector_store.search(request.prompt, top_k=3)
            context_str = ""
            for res in results:
                context_str += f"{res['content']}\n\n"
            
            if context_str:
                final_prompt = (
                    "Use the following pieces of context to answer the question at the end.\n"
                    "If you don't know the answer, just say that you don't know.\n\n"
                    f"Context:\n{context_str}\n"
                    f"Question: {request.prompt}\n"
                    "Answer:"
                )
                
        # Encode the prompt
        if final_prompt:
            prompt_ids = tokenizer.encode(final_prompt)
            context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            
        print(f"DEBUG: context device is {context.device}")
        
        # Generate
        with torch.no_grad(): # good practice to use no_grad for inference
            if request.use_speculative_decoding and draft_model is not None:
                generated_indices = speculative_decode(
                    model, draft_model, context, tokenizer,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature
                )[0].tolist()
            else:
                generated_indices = model.generate(
                    context, 
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    repetition_penalty=request.repetition_penalty
                )[0].tolist()
            
        # Extract only the newly generated text for RAG to look clean
        if request.use_rag and final_prompt != request.prompt:
            new_indices = generated_indices[len(prompt_ids):]
            generated_text = tokenizer.decode(new_indices)
        else:
            generated_text = tokenizer.decode(generated_indices)
        
        return GenerateResponse(prompt=request.prompt, generated_text=generated_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

@app.post("/generate_stream")
def generate_text_stream(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet or failed to load.")
        
    try:
        final_prompt = request.prompt
        
        # 1. RAG Context Retrieval
        if request.use_rag and vector_store is not None and request.prompt:
            results = vector_store.search(request.prompt, top_k=3)
            context_str = ""
            for res in results:
                context_str += f"{res['content']}\n\n"
            
            if context_str:
                final_prompt = (
                    "Use the following pieces of context to answer the question at the end.\n"
                    "If you don't know the answer, just say that you don't know.\n\n"
                    f"Context:\n{context_str}\n"
                    f"Question: {request.prompt}\n"
                    "Answer:"
                )
                
        if final_prompt:
            prompt_ids = tokenizer.encode(final_prompt)
            context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            
        def event_stream():
            for idx_next in model.generate_stream(
                context, 
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty
            ):
                word = tokenizer.decode([idx_next.item()])
                yield f"data: {word}\n\n"
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating stream: {str(e)}")

@app.post("/agent_chat")
def agent_chat_endpoint(request: AgentRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet or failed to load.")
        
    tools = [
        Tool("Calculator", "Evaluates basic math expressions", evaluate_math),
        Tool("Wikipedia", "Searches Wikipedia for a given query", search_wikipedia),
        Tool("PythonREPL", "Executes Python code and returns output", execute_python)
    ]
    agent = Agent(model, tokenizer, device, tools=tools)
    response = agent.run(request.prompt)
    return {"response": response}

# This allows running the file directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
