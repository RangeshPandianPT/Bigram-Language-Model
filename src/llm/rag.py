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

