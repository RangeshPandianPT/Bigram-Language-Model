from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn
from contextlib import asynccontextmanager

from scripts._bootstrap import ROOT_DIR
from llm.tokenizer import BPETokenizer
from llm.config import GPTConfig, TrainConfig
from llm.model import GPTLanguageModel
from llm.paths import MODEL_PATH, TOKENIZER_PREFIX

# Global variables to hold our model and tokenizer
model = None
tokenizer = None
device = None

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
        model_path = str(MODEL_PATH)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
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
        "tokenizer_loaded": tokenizer is not None,
        "device": str(device) if device is not None else None,
    }

@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet or failed to load.")
    
    try:
        # Encode the prompt
        if request.prompt:
            prompt_ids = tokenizer.encode(request.prompt)
            context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            
        print(f"DEBUG: context device is {context.device}")
        
        # Generate
        with torch.no_grad(): # good practice to use no_grad for inference
            generated_indices = model.generate(
                context, 
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty
            )[0].tolist()
            
        generated_text = tokenizer.decode(generated_indices)
        
        return GenerateResponse(prompt=request.prompt, generated_text=generated_text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

# This allows running the file directly
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
