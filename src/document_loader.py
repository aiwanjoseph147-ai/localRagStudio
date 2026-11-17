"""
Document Loading and Preprocessing Module.
Handles loading various file types (PDF, TXT, MD, JSON, CSV) and web URLs
into normalized LangChain Document objects.
"""

import os
import re
import tempfile
from typing import List, Tuple, Union
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from .config import SUPPORTED_EXTENSIONS


class DocumentManager:
    """Manages reading and parsing document files and URLs into LangChain Document instances."""

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

    @classmethod
    def load_url(cls, url: str) -> Tuple[List[Document], str]:
        """
        NEW FEATURE: Fetches and parses a web URL (HTML page or online PDF).
        Returns:
            (documents, page_title)
        """
        url = url.strip()
        if not re.match(r"^https?://", url, re.IGNORECASE):
            url = "https://" + url

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8"
        }

        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
        except Exception as e:
            raise ValueError(f"Failed to fetch URL ({url}): {e}")

        content_type = resp.headers.get("Content-Type", "").lower()

        # Handle online PDF
        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = tmp.name

            try:
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            title = os.path.basename(url.split("?")[0]) or "Online PDF Document"
            for doc in docs:
                doc.metadata["file_name"] = f"[PDF URL] {title}"
                doc.metadata["url"] = url
                if "page" in doc.metadata and isinstance(doc.metadata["page"], int):
                    doc.metadata["page"] += 1
                else:
                    doc.metadata["page"] = 1
            return docs, title

        # Handle HTML web page
        soup = BeautifulSoup(resp.text, "html.parser")

        # Strip unneeded elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
            tag.decompose()

        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()
        if not title:
            title = url

        # Extract main text
        lines = []
        for elem in soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "article", "blockquote"]):
            text = elem.get_text().strip()
            if text and len(text) > 15:
                lines.append(text)

        full_text = "\n\n".join(lines)
        if not full_text.strip():
            # Fallback to whole body text
            full_text = "\n".join([line.strip() for line in soup.get_text().splitlines() if line.strip()])

        if not full_text.strip():
            raise ValueError(f"No readable content could be extracted from {url}")

        doc = Document(
            page_content=full_text,
            metadata={
                "source": url,
                "file_name": f"[Web] {title[:60]}",
                "title": title,
                "url": url,
                "page": 1
            }
        )
        return [doc], title
