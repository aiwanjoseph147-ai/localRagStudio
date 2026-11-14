# 🔍 Local RAG Studio (Ollama + FAISS)

A modular, production-ready **Retrieval-Augmented Generation (RAG)** application built in Python using **LangChain**, **Ollama**, and **FAISS**.

Ingest **any set of documents** (PDFs, TXT, Markdown, CSV), chunk and embed them locally, generate **one-click Executive Briefings**, and ask questions with strict fact-grounding, conversation history, and clickable citations with **relevance match percentages**.

---

## 🌟 Key Features

1. **Multi-Format Ingestion**:
   - Supports uploading or pointing to sets of files (`.pdf`, `.txt`, `.md`, `.json`, `.csv`).
   - Automatically parses pages and extracts source metadata.
   - Built-in one-click support for the included `./data` folder (10 Elrond Investigation PDFs).

2. **📑 Executive Briefing & Smart Question Generator (NEW)**:
   - One-click synthesis of the entire document collection.
   - Generates an **Executive Overview**, **Key Findings & Evidence**, and **Key Entities / Topics**.
   - Automatically suggests **clickable deep-dive questions** tailored to the specific documents loaded.

3. **🎯 Relevance Scoring & Source Attribution (NEW)**:
   - Computes normalized vector similarity match percentages (`Match: 88.5%`) for every retrieved chunk.
   - Every answer includes expandable citations indicating exact source filenames, page numbers, and chunk previews.

4. **FAISS Vector Indexing & Persistence**:
   - High-speed vector similarity search using `bge-m3` embeddings.
   - Save indexes to disk (`save_local`) and reload them later without re-embedding (`load_local`).

5. **Conversational Memory & Grounding**:
   - Multi-turn conversation tracking (`chat_history`) for contextual follow-up questions.
   - Grounded persona instructions (AI Detective, Assistant, Strict Fact-Checker, Academic) to prevent hallucinations.

6. **Multiple User Interfaces**:
   - **Modern Web Studio (`app.py`)**: Zero-dependency local browser interface with drag-and-drop upload, sliders, and chat UI.
   - **Interactive CLI (`cli.py`)**: Fast terminal interface with special slash commands (`/summary`, `/sources`, `/stats`).
   - **Streamlit App (`streamlit_app.py`)**: Optional Streamlit dashboard interface.

---

## 🏗️ Architecture & Workflow

```
 📄 User Documents (PDF / TXT / MD)
                │
                ▼
      [ DocumentManager ]
     (Loads & extracts pages)
                │
                ▼
   [ VectorStoreManager: Chunking ]
   (RecursiveCharacterTextSplitter)
      chunk_size=500, overlap=150
                │
                ▼
   [ OllamaEmbeddings: bge-m3 ]
                │
                ▼
      [ FAISS Vector Store ]  ◄── (save_local / load_local)
                │
     ┌──────────┴──────────┐
     ▼                     ▼
[ User Question ]    [ Top-K Chunks with Relevance % ]
     │                     │
     └──────────┬──────────┘
                ▼
    [ Grounded Prompt Builder ]
    (Context + History + Persona)
                │
                ▼
     [ ChatOllama: qwen2.5:3b ]
                │
                ▼
    🎯 Answer with Relevance Citations
    💡 Dynamic Exploration Questions
```

---

## 📁 Modular Project Structure

```
rag_ollama/
├── src/                          # Core application package
│   ├── __init__.py               # Package exports (RAGPipeline, DocumentManager, etc.)
│   ├── config.py                 # Defaults, hyperparameters, and personas
│   ├── document_loader.py        # Multi-format document loading & metadata parsing
│   ├── vector_store.py           # Text chunking, FAISS index, and relevance search
│   ├── models.py                 # Ollama LLM and embedding utilities
│   └── pipeline.py               # RAGPipeline orchestrator, Q&A, and Executive Briefing
│
├── web/                          # Frontend assets
│   └── index.html                # Modern single-page chat & document studio UI
│
├── data/                         # Sample dataset (Elrond Investigation PDF files)
├── elrond_investigation/         # Pre-computed FAISS vector database
├── notebooks/                    # Reference research & tuning notebooks
│   ├── Elrond_Investigation_RAG_Pipeline.ipynb
│   └── RAG_Hyperparameter_Tuning.ipynb
│
├── app.py                        # Standalone Web Application server (zero dependencies)
├── cli.py                        # Interactive Command-Line Interface (CLI)
├── streamlit_app.py              # Optional Streamlit dashboard
├── rag_pipeline.py               # Backward-compatibility bridge
├── run_app.bat                   # 1-Click launcher for Web App (Windows)
├── run_cli.bat                   # 1-Click launcher for CLI (Windows)
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## ⚡ Prerequisites

1. **Install and run [Ollama](https://ollama.com/)**:
   ```bash
   ollama --version
   ```

2. **Pull the recommended models**:
   ```bash
   ollama pull qwen2.5:3b
   ollama pull bge-m3
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Application

### 1. Web Studio (Recommended)

Launch the web application with zero external web dependencies:

```bash
..\.venv\Scripts\python.exe app.py
```
*(Or simply double-click `run_app.bat` on Windows)*

Open your browser to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

**Web UI Features**:
- **Drag-and-Drop Ingestion**: Upload custom PDF, TXT, or Markdown documents with live progress.
- **📑 Executive Briefing**: Click the new **"Generate Executive Briefing"** button to summarize the entire knowledge base and generate clickable questions.
- **Inspectable Sources**: Expandable accordion card under every answer displaying the file name, page number, chunk text, and **Relevance Match %**.
- **Interactive Tuning**: Sliders for `Chunk Size` (200–1500), `Chunk Overlap` (0–400), and `Top-K Retrieved Chunks` (1–15).
- **Persona Switcher**: Choose between **Helpful Assistant**, **AI Detective**, **Strict Fact-Checker**, or **Academic Researcher**.

---

### 2. Interactive Terminal CLI

Run the full pipeline directly in your terminal:

```bash
..\.venv\Scripts\python.exe cli.py
```
*(Or double-click `run_cli.bat` on Windows)*

**CLI Commands**:
- `/summary` — **(NEW)** Generate an Executive Briefing across documents with curated questions.
- `/sources` — Show retrieved chunks with **Relevance Match %** and page numbers.
- `/stats` — Display active vector index metrics and hyperparameters.
- `/persona <type>` — Switch persona (`assistant`, `detective`, `strict_analyst`, `academic`).
- `/save <dir>` — Persist the current FAISS vector store to disk.
- `/load <dir>` — Load an existing FAISS vector store from disk.
- `/clear` — Reset conversational memory.
- `exit` — Exit the CLI.

---

### 3. Streamlit Dashboard

If you prefer Streamlit:

```bash
..\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

---

## 💻 Python API Usage

```python
from src import RAGPipeline

# 1. Initialize RAG Pipeline
rag = RAGPipeline(
    llm_model="qwen2.5:3b",
    embed_model="bge-m3",
    chunk_size=500,
    chunk_overlap=150,
    k=5,
    persona="assistant"
)

# 2. Ingest documents (files or directory)
stats = rag.build_index_from_files("./data")
print(f"Indexed {stats['files_count']} files into {stats['total_chunks']} chunks.")

# 3. Generate Executive Briefing & Smart Questions (NEW)
briefing = rag.generate_summary()
print("Executive Briefing:\n", briefing["summary"])
print("Suggested Questions:\n", briefing["suggested_questions"])

# 4. Ask questions with conversational memory & relevance scores
result = rag.ask("Why is Elrond under investigation?")
print("Answer:\n", result["answer"])

# 5. Inspect retrieved sources with relevance %
for source in result["sources"]:
    print(f"- {source['file_name']} (Page {source['page']}) -> Match: {source['relevance_score']}")

# 6. Save index for reuse later
rag.save_index("my_faiss_index")
```

---

## ⚙️ Hyperparameter Tuning Guide

| Parameter | Default | Range | Description |
|---|---|---|---|
| `chunk_size` | `500` | 200 – 1500 | Character length of each chunk. Smaller chunks isolate specific facts; larger chunks provide broader context. |
| `chunk_overlap` | `150` | 50 – 300 | Characters shared between adjacent chunks to prevent cutting sentences across boundaries. |
| `k` | `5` | 1 – 15 | Number of highest-similarity chunks retrieved from FAISS and fed into the LLM context. |
| `persona` | `assistant` | detective / analyst / academic | Controls the system prompt and tone while enforcing factual grounding. |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
