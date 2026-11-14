"""
Ollama Model Utilities and Initializers.
"""

from typing import List, Optional
from langchain_ollama import ChatOllama, OllamaEmbeddings
import ollama


def get_llm(model_name: str) -> ChatOllama:
    """Factory function for Ollama Chat Model."""
    return ChatOllama(model=model_name)


def get_embeddings(model_name: str) -> OllamaEmbeddings:
    """Factory function for Ollama Embeddings Model."""
    return OllamaEmbeddings(model=model_name)


def list_ollama_models() -> List[str]:
    """Helper to list available models installed locally in Ollama."""
    try:
        models_data = ollama.list()
        if hasattr(models_data, "models"):
            return [m.model for m in models_data.models]
        elif isinstance(models_data, dict) and "models" in models_data:
            return [m["name"] for m in models_data["models"]]
    except Exception as e:
        print(f"Notice: Could not query Ollama models directly ({e})")
    return ["qwen2.5:3b", "qwen2.5:1.5b", "bge-m3:latest"]
