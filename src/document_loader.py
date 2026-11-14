"""
Document Loading and Preprocessing Module.
Handles loading various file types into normalized LangChain Document objects.
"""

import os
from typing import List, Tuple, Union
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from .config import SUPPORTED_EXTENSIONS


class DocumentManager:
    """Manages reading and parsing document files into LangChain Document instances."""

    @classmethod
    def load_file(cls, file_path: str) -> List[Document]:
        """Loads a single file into a list of Document objects."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        docs: List[Document] = []

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
        elif ext in {".txt", ".md", ".json", ".csv"}:
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
        else:
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            except Exception:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                docs = [Document(page_content=content, metadata={"source": file_path, "page": 1})]

        base_name = os.path.basename(file_path)
        for doc in docs:
            doc.metadata["file_name"] = base_name
            if "page" not in doc.metadata:
                doc.metadata["page"] = 1
            else:
                try:
                    if isinstance(doc.metadata["page"], int):
                        doc.metadata["page"] += 1
                except Exception:
                    pass
        return docs

    @classmethod
    def load_files_or_directory(cls, path_or_paths: Union[str, List[str]]) -> Tuple[List[Document], List[str]]:
        """
        Loads documents from a directory path, a single file path, or a list of file paths.
        Returns:
            all_documents: List of LangChain Document objects
            loaded_files: List of file paths successfully processed
        """
        file_paths = []
        if isinstance(path_or_paths, str):
            if os.path.isdir(path_or_paths):
                for root, _, files in os.walk(path_or_paths):
                    for file in sorted(files):
                        ext = os.path.splitext(file)[1].lower()
                        if ext in SUPPORTED_EXTENSIONS:
                            file_paths.append(os.path.join(root, file))
            elif os.path.isfile(path_or_paths):
                file_paths.append(path_or_paths)
        elif isinstance(path_or_paths, list):
            for p in path_or_paths:
                if os.path.isdir(p):
                    sub_docs, sub_paths = cls.load_files_or_directory(p)
                    file_paths.extend(sub_paths)
                elif os.path.isfile(p):
                    file_paths.append(p)

        file_paths = sorted(list(set(file_paths)))
        all_docs: List[Document] = []
        loaded_files: List[str] = []

        for fp in file_paths:
            try:
                docs = cls.load_file(fp)
                all_docs.extend(docs)
                loaded_files.append(fp)
            except Exception as e:
                print(f"Warning: Failed to load {fp}: {e}")

        return all_docs, loaded_files
