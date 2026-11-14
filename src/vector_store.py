"""
FAISS Vector Store and Text Chunking Management.
"""

import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


class VectorStoreManager:
    """Manages document chunking, FAISS index construction, persistence, and retrieval."""

    @staticmethod
    def chunk_documents(
        documents: List[Document],
        chunk_size: int = 500,
        chunk_overlap: int = 150
    ) -> List[Document]:
        """Splits raw documents into smaller overlapping text chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_documents(documents)

    @staticmethod
    def create_vector_store(
        chunks: List[Document],
        embeddings: OllamaEmbeddings
    ) -> FAISS:
        """Embeds text chunks and builds an in-memory FAISS vector index."""
        return FAISS.from_documents(chunks, embeddings)

    @staticmethod
    def search_with_relevance(
        vector_db: FAISS,
        query: str,
        k: int = 5
    ) -> List[Tuple[Document, float]]:
        """
        Performs similarity search with FAISS and returns documents with
        normalized relevance percentage (0-100%).
        """
        docs_and_scores = vector_db.similarity_search_with_score(query, k=k)
        results = []
        for doc, distance in docs_and_scores:
            # Distance in FAISS L2 typically ranges from 0 (identical) to ~2
            # Map distance to an intuitive percentage
            norm_score = max(5.0, min(99.0, round((1.0 / (1.0 + float(distance))) * 100, 1)))
            results.append((doc, norm_score))
        return results

    @staticmethod
    def save_vector_store(vector_db: FAISS, directory_path: str) -> str:
        """Persists the FAISS index to disk."""
        os.makedirs(directory_path, exist_ok=True)
        vector_db.save_local(directory_path)
        return directory_path

    @staticmethod
    def load_vector_store(directory_path: str, embeddings: OllamaEmbeddings) -> FAISS:
        """Loads a persisted FAISS vector index from disk."""
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Index directory does not exist: {directory_path}")

        return FAISS.load_local(
            directory_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
