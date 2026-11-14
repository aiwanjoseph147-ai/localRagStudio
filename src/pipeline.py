"""
End-to-End RAG Pipeline Orchestrator.
Combines document loading, chunking, embeddings, FAISS retrieval, relevance scoring,
executive document summarization, and Ollama LLM answering.
"""

import os
import re
from typing import List, Dict, Any, Optional, Union

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama

from .config import RAGConfig, PERSONAS
from .document_loader import DocumentManager
from .vector_store import VectorStoreManager
from .models import get_llm, get_embeddings


class RAGPipeline:
    """
    Main RAG Pipeline class.
    Coordinates document ingestion, vector database management, relevance scoring,
    executive summarization, and conversational Q&A.
    """

    def __init__(
        self,
        llm_model: str = RAGConfig.default_llm_model,
        embed_model: str = RAGConfig.default_embed_model,
        chunk_size: int = RAGConfig.chunk_size,
        chunk_overlap: int = RAGConfig.chunk_overlap,
        k: int = RAGConfig.k,
        persona: str = RAGConfig.default_persona,
        custom_system_prompt: Optional[str] = None
    ):
        self.llm_model_name = llm_model
        self.embed_model_name = embed_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.persona_name = persona
        self.custom_system_prompt = custom_system_prompt

        self._llm: Optional[ChatOllama] = None
        self._embeddings: Optional[OllamaEmbeddings] = None
        self.vector_db: Optional[FAISS] = None

        self.indexed_files: List[str] = []
        self.total_pages: int = 0
        self.total_chunks: int = 0
        self.chat_history: List[Dict[str, str]] = []

    @property
    def embeddings(self) -> OllamaEmbeddings:
        """Cached OllamaEmbeddings instance."""
        if self._embeddings is None:
            self._embeddings = get_embeddings(self.embed_model_name)
        return self._embeddings

    @property
    def llm(self) -> ChatOllama:
        """Cached ChatOllama instance."""
        if self._llm is None:
            self._llm = get_llm(self.llm_model_name)
        return self._llm

    def update_models(self, llm_model: Optional[str] = None, embed_model: Optional[str] = None):
        """Update model names and invalidate cached instances."""
        if llm_model and llm_model != self.llm_model_name:
            self.llm_model_name = llm_model
            self._llm = None
        if embed_model and embed_model != self.embed_model_name:
            self.embed_model_name = embed_model
            self._embeddings = None

    def update_hyperparameters(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        k: Optional[int] = None,
        persona: Optional[str] = None,
        custom_system_prompt: Optional[str] = None
    ):
        """Update tuning parameters dynamically."""
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
        if k is not None:
            self.k = k
        if persona is not None:
            self.persona_name = persona
        if custom_system_prompt is not None:
            self.custom_system_prompt = custom_system_prompt

    def build_index_from_files(
        self,
        path_or_paths: Union[str, List[str]],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Loads document files or directories, splits into chunks, and indexes into FAISS.
        """
        c_size = chunk_size if chunk_size is not None else self.chunk_size
        c_overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap

        # 1. Load documents
        raw_docs, loaded_files = DocumentManager.load_files_or_directory(path_or_paths)
        if not raw_docs:
            raise ValueError(f"No valid documents found at: {path_or_paths}")

        # 2. Text chunking
        chunks = VectorStoreManager.chunk_documents(
            documents=raw_docs,
            chunk_size=c_size,
            chunk_overlap=c_overlap
        )

        # 3. Embedding and FAISS indexing
        vector_db = VectorStoreManager.create_vector_store(chunks, self.embeddings)

        self.vector_db = vector_db
        self.indexed_files = loaded_files
        self.total_pages = len(raw_docs)
        self.total_chunks = len(chunks)
        self.chunk_size = c_size
        self.chunk_overlap = c_overlap

        return {
            "status": "success",
            "files_count": len(loaded_files),
            "files": [os.path.basename(f) for f in loaded_files],
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "chunk_size": c_size,
            "chunk_overlap": c_overlap,
        }

    def save_index(self, directory_path: str) -> str:
        """Saves current FAISS index to disk."""
        if self.vector_db is None:
            raise ValueError("No active index to save.")
        return VectorStoreManager.save_vector_store(self.vector_db, directory_path)

    def load_index(self, directory_path: str) -> Dict[str, Any]:
        """Loads a saved FAISS index from disk."""
        self.vector_db = VectorStoreManager.load_vector_store(directory_path, self.embeddings)
        return {
            "status": "success",
            "index_path": directory_path
        }

    def get_system_prompt(self) -> str:
        """Returns the active grounding persona instruction."""
        if self.custom_system_prompt and self.custom_system_prompt.strip():
            return self.custom_system_prompt.strip()
        return PERSONAS.get(self.persona_name, PERSONAS["assistant"])

    def ask(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        k: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes grounded question answering with relevance scoring:
        1. Retrieves top-k chunks with similarity scores from FAISS.
        2. Formats context with citation labels and relevance percentages.
        3. Formulates grounded prompt with conversational history.
        4. Invokes LLM.
        5. Updates chat history and returns answer with source metadata.
        """
        if self.vector_db is None:
            raise ValueError("Knowledge base is empty. Please load or index documents first.")

        top_k = k if k is not None else self.k

        # 1. Retrieve chunks with relevance scoring
        retrieved_with_scores = VectorStoreManager.search_with_relevance(
            self.vector_db, question, k=top_k
        )

        # 2. Build context string and source metadata
        context_parts = []
        sources = []
        for i, (chunk, relevance) in enumerate(retrieved_with_scores, 1):
            source_file = chunk.metadata.get("file_name") or os.path.basename(chunk.metadata.get("source", "Unknown"))
            page_num = chunk.metadata.get("page", 1)
            content = chunk.page_content.strip()

            context_parts.append(
                f"[Source {i}: {source_file} (Page {page_num}) - Relevance {relevance}%]\n{content}"
            )
            sources.append({
                "chunk_id": i,
                "file_name": source_file,
                "page": page_num,
                "content": content,
                "relevance_score": f"{relevance}%",
                "relevance_num": relevance
            })

        context_str = "\n\n".join(context_parts)

        # 3. Format conversational history
        history = chat_history if chat_history is not None else self.chat_history
        history_str = ""
        if history:
            for turn in history:
                role = "User" if turn.get("role") in ("user", "human") else "Assistant"
                history_str += f"{role}: {turn.get('content', '')}\n\n"

        # 4. Construct prompt
        active_sys_prompt = system_prompt or self.get_system_prompt()
        prompt_text = f"""{active_sys_prompt}

Chat History:
{history_str if history_str.strip() else "None"}

Context:
{context_str}

Question:
{question}
"""

        # 5. Invoke Chat LLM
        response = self.llm.invoke(prompt_text)
        answer_text = response.content

        # 6. Update internal memory
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer_text})

        return {
            "question": question,
            "answer": answer_text,
            "sources": sources,
            "context_length": len(context_str),
            "chunks_retrieved": len(retrieved_with_scores),
            "model_used": self.llm_model_name
        }

    def generate_summary(self, max_chunks: int = 6) -> Dict[str, Any]:
        """
        NEW FEATURE: Executive Document Briefing & Smart Question Generator.
        Analyzes the active knowledge base and generates:
        1. An Executive Overview
        2. Key Findings & Core Evidence
        3. Notable Entities / Topics
        4. Curated Deep-Dive Questions for one-click exploration
        """
        if self.vector_db is None:
            raise ValueError("No active index to summarize. Please load documents first.")

        # Search across broad discovery queries to capture multi-angle context
        broad_terms = "overview summary background investigation allegations report evidence conclusions"
        retrieved_with_scores = VectorStoreManager.search_with_relevance(
            self.vector_db, broad_terms, k=max_chunks
        )

        context_parts = []
        sources = []
        for i, (chunk, relevance) in enumerate(retrieved_with_scores, 1):
            source_file = chunk.metadata.get("file_name") or os.path.basename(chunk.metadata.get("source", "Unknown"))
            page_num = chunk.metadata.get("page", 1)
            content = chunk.page_content.strip()

            context_parts.append(f"[Document Excerpt {i}: {source_file} (Page {page_num})]\n{content}")
            sources.append({
                "chunk_id": i,
                "file_name": source_file,
                "page": page_num,
                "content": content,
                "relevance_score": f"{relevance}%"
            })

        context_str = "\n\n".join(context_parts)

        prompt = f"""
You are an expert intelligence and document analyst.
Based on the following excerpts from the loaded document archive, create a structured Executive Briefing.

Your output MUST include these exact sections:
### 📌 Executive Overview
(2-3 paragraphs synthesizing what these documents cover and their primary subject matter)

### 🔍 Key Findings & Evidence
(Bulleted list of the most critical facts, testimonies, or assertions)

### 👥 Key Entities & Topics Involved
(Summary of prominent figures, groups, or subjects)

### 💡 Recommended Deep-Dive Questions
Provide exactly 4 insightful questions that a user should investigate based on these files.
Format each question on its own line prefixed with:
> [Question] <your question here>

Context:
{context_str}
"""

        response = self.llm.invoke(prompt)
        raw_text = response.content

        # Parse the recommended questions
        suggested_questions = []
        for line in raw_text.splitlines():
            line_clean = line.strip()
            match = re.search(r">\s*\[Question\]\s*(.+)", line_clean, re.IGNORECASE)
            if match:
                suggested_questions.append(match.group(1).strip())
            elif line_clean.startswith(("- Question:", "* Question:", "1. Question:")):
                q = re.sub(r"^[-*0-9.]+\s*Question:\s*", "", line_clean).strip()
                if q:
                    suggested_questions.append(q)

        # Fallback default questions if regex didn't catch 4
        if len(suggested_questions) < 2:
            suggested_questions = [
                "What are the main allegations or primary claims made in these documents?",
                "What specific evidence or witness testimonies support these claims?",
                "Who are the key figures involved and what are their respective statements?",
                "What conclusions or official statuses are documented?"
            ]

        # Add briefing to chat memory so user can ask follow-ups naturally
        self.chat_history.append({"role": "user", "content": "Generate an Executive Briefing of the documents."})
        self.chat_history.append({"role": "assistant", "content": raw_text})

        return {
            "summary": raw_text,
            "suggested_questions": suggested_questions[:4],
            "sources": sources,
            "files_analyzed": len(set(s["file_name"] for s in sources))
        }

    def clear_chat_history(self):
        """Clears stored conversational memory."""
        self.chat_history = []

    def get_stats(self) -> Dict[str, Any]:
        """Returns metadata about the active pipeline."""
        return {
            "has_index": self.vector_db is not None,
            "files_indexed": [os.path.basename(f) for f in self.indexed_files],
            "total_files": len(self.indexed_files),
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.k,
            "llm_model": self.llm_model_name,
            "embed_model": self.embed_model_name,
            "persona": self.persona_name,
            "history_turns": len(self.chat_history) // 2
        }
