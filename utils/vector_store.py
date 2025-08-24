import faiss
import numpy as np
import torch
from utils.embedding_handler import load_embedding_model


def create_vector_store(embeddings: torch.Tensor) -> faiss.IndexFlatIP:
    """Create FAISS index from embeddings"""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.cpu().numpy())
    return index

def search_similar(index: faiss.IndexFlatIP, query: str, k: int = 3) -> list:
    """Search for similar documents using optimized similarity search"""
    model = load_embedding_model()
    query_embedding = model.encode([query], convert_to_tensor=True)
    distances, indices = index.search(query_embedding.cpu().numpy(), k)
    return indices[0].tolist()
