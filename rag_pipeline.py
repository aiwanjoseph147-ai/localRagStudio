"""
Backward-compatibility bridge for rag_pipeline.
Directly re-exports the modular components from the src package.
"""

from src import (
    RAGPipeline,
    DocumentManager,
    VectorStoreManager,
    RAGConfig,
    PERSONAS,
    SUPPORTED_EXTENSIONS,
    get_llm,
    get_embeddings,
    list_ollama_models
)

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
