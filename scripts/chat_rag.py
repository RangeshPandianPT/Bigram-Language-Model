import os
import torch
import argparse

from scripts._bootstrap import ROOT_DIR
from llm.tokenizer import BPETokenizer
from llm.config import GPTConfig, TrainConfig
from llm.model import GPTLanguageModel
from llm.paths import MODEL_PATH, TOKENIZER_PREFIX
from llm.rag import DocumentLoader, TextChunker, VectorStore

def main():
    parser = argparse.ArgumentParser(description='RAG Chat with trained LLM')
    parser.add_argument('--model_path', type=str, default=str(MODEL_PATH), help='Path to model checkpoint')
    parser.add_argument('--doc_dir', type=str, default=os.path.join(ROOT_DIR, 'data', 'documents'), help='Directory containing documents to ingest')
    parser.add_argument('--max_tokens', type=int, default=300, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling')
    parser.add_argument('--top_k', type=int, default=3, help='How many document chunks to retrieve')
    args = parser.parse_args()

    # 1. Initialize RAG system
    print(f"Loading documents from {args.doc_dir}...")
    loader = DocumentLoader()
    docs = loader.load_directory(args.doc_dir)
    
    if not docs:
        print(f"Warning: No documents found in {args.doc_dir}. RAG will have no context. Please add .txt files.")
    
    chunker = TextChunker()
    chunks = []
    for doc in docs:
        chunks.extend(chunker.chunk_document(doc))
        
    print(f"Created {len(chunks)} text chunks.")
    
    print("Initializing VectorStore (loading embedding model)...")
    try:
        vector_store = VectorStore()
        vector_store.ingest(chunks)
    except ImportError as e:
        print(f"Error: {e}")
        return

    # 2. Load LLM
    print("Loading language model...")
    train_config = TrainConfig()
    gpt_config = GPTConfig()
    
    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PREFIX))
    gpt_config.vocab_size = len(tokenizer.vocab)
    
    model = GPTLanguageModel(gpt_config)
    model.load_state_dict(torch.load(args.model_path, map_location=train_config.device))
    model.to(train_config.device)
    model.eval()
    print("LLM loaded and ready.")
    
    print("="*80)
    print("RAG Chat Initialized. Type 'quit' or 'exit' to stop.")
    print("="*80)
    
    # 3. Interactive Loop
    while True:
        try:
            query = input("\nUser: ")
            if query.lower() in ['quit', 'exit']:
                break
            if not query.strip():
                continue
                
            # Retrieve Context
            results = vector_store.search(query, top_k=args.top_k)
            
            context_str = ""
            if results:
                print("\n[Retrieved Context]")
                for i, res in enumerate(results):
                    snippet = res['content'][:100].replace('\n', ' ') + "..."
                    print(f" - [{res['source']}] {snippet} (dist: {res['distance']:.2f})")
                    context_str += f"{res['content']}\n\n"
            else:
                print("\n[No context retrieved]")
            
            # Construct Prompt
            if context_str:
                prompt = (
                    "Use the following pieces of context to answer the question at the end.\n"
                    "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
                    f"Context:\n{context_str}\n"
                    f"Question: {query}\n"
                    "Answer:"
                )
            else:
                prompt = f"Question: {query}\nAnswer:"
                
            prompt_ids = tokenizer.encode(prompt)
            context_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=train_config.device)
            
            # Generate
            print(f"\nAssistant: ", end="", flush=True)
            generated_indices = model.generate(
                context_tensor, 
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )[0].tolist()
            
            # We only want to print the newly generated part
            new_indices = generated_indices[len(prompt_ids):]
            response = tokenizer.decode(new_indices)
            print(response.strip())
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
