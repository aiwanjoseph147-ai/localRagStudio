"""
RAG Package Initialization.
Exposes core classes and helper functions.
"""

from .config import RAGConfig, PERSONAS, SUPPORTED_EXTENSIONS
from .document_loader import DocumentManager
from .vector_store import VectorStoreManager
from .models import get_llm, get_embeddings, list_ollama_models
from .pipeline import RAGPipeline

__all__ = [
    "RAGPipeline",
    "DocumentManager",
    "VectorStoreManager",
    "RAGConfig",
    "PERSONAS",
    "SUPPORTED_EXTENSIONS",
    "get_llm",
    "get_embeddings",
    "list_ollama_models"
]
