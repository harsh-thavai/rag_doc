from sentence_transformers import SentenceTransformer
import torch
import streamlit as st

# Initialize model once (cached)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-mpnet-base-v2', device='cpu')

def generate_embeddings(texts: list) -> torch.Tensor:
    """Generate embeddings for text chunks using high-quality model"""
    model = load_embedding_model()
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=True)