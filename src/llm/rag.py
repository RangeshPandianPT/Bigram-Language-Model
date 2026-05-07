import os
import re
from typing import List, Dict, Any

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Warning: faiss or sentence_transformers not installed. RAG features will be disabled.")
    faiss = None
    np = None
    SentenceTransformer = None

class DocumentLoader:
    """Loads documents from a given directory."""
    @staticmethod
    def load_directory(dir_path: str) -> List[Dict[str, str]]:
        documents = []
        if not os.path.exists(dir_path):
            return documents
        
        for filename in os.listdir(dir_path):
            if filename.endswith(".txt") or filename.endswith(".md"):
                file_path = os.path.join(dir_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    documents.append({
                        "id": filename,
                        "content": content
                    })
        return documents

class TextChunker:
    """Splits text into chunks with overlap to preserve context."""
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Dict[str, str]) -> List[Dict[str, str]]:
        text = document["content"]
        # Basic clean up
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to snap to the nearest space or punctuation if not at the very end
            if end < len(text):
                # Search backwards for a space or punctuation
                match = re.search(r'[.!?\s]', text[end:start:-1])
                # Only snap back if we don't snap back too far (e.g. more than half the chunk)
                if match and match.start() < (self.chunk_size // 2):
                    end = end - match.start()
                    
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "id": f"{document['id']}_chunk_{len(chunks)}",
                    "content": chunk_text,
                    "source": document['id']
                })
            
            # Advance start pointer, accounting for overlap
            next_start = end - self.chunk_overlap
            
            # Prevent infinite loop if no progress is made
            if next_start <= start:
                start += self.chunk_size
            else:
                start = next_start
                
        return chunks

