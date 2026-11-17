#!/usr/bin/env python3
"""
Interactive Command Line Interface for RAG Question Answering
Powered by LangChain, Ollama, and FAISS.
Features:
- Document & Folder Ingestion (PDF, TXT, MD, CSV)
- Web Link Ingestion & Instant Summarization
- Continuous Conversational Q&A
- Relevance Scoring & Source Inspection
"""

import os
import sys
import argparse
from rag_pipeline import RAGPipeline, DocumentManager, list_ollama_models

BANNER = """
================================================================================
                    RAG DOCUMENT & URL INVESTIGATOR & QA APP
          Retrieval-Augmented Generation with Ollama & FAISS
================================================================================
"""

HELP_TEXT = """
Available Commands:
  /summary        - Generate an Executive Briefing and Smart Questions from the active content
  /url <url>      - Ingest a web page or online PDF URL, summarize it, and enable Q&A
  /sources        - Show sources & chunks used for the previous answer with relevance scores
  /stats          - Display current index and hyperparameter statistics
  /clear          - Reset conversation history
  /persona <type> - Switch persona (detective, assistant, strict_analyst, academic)
  /save <dir>     - Save the active FAISS vector store to disk
  /load <dir>     - Load a saved FAISS index from disk
  /index <path>   - Index a new file or directory of documents
  /help           - Display this help message
  exit or quit    - Exit the program
"""


def format_sources(sources):
    if not sources:
        return "No sources available."
    out = []
    out.append("\n" + "=" * 45 + " RETRIEVED SOURCES & RELEVANCE " + "=" * 45)
    for s in sources:
        rel = s.get("relevance_score", "N/A")
        out.append(f"\n[Source #{s['chunk_id']}] File: {s['file_name']} | Page: {s['page']} | Match: {rel}")
        out.append("-" * 75)
        out.append(s['content'])
    out.append("=" * 122 + "\n")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="RAG Document & URL QA CLI")
    parser.add_argument("--dir", "-d", type=str, help="Path to directory containing documents to index (e.g. ./data)")
    parser.add_argument("--files", "-f", nargs="+", help="Specific files to index")
    parser.add_argument("--url", "-u", type=str, help="Web URL or online PDF link to ingest and query")
    parser.add_argument("--load-index", "-i", type=str, help="Path to pre-existing FAISS vector store (e.g. elrond_investigation)")
    parser.add_argument("--llm", type=str, default="qwen2.5:3b", help="Ollama LLM model name (default: qwen2.5:3b)")
    parser.add_argument("--embed", type=str, default="bge-m3", help="Ollama embedding model name (default: bge-m3)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters (default: 500)")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap in characters (default: 150)")
    parser.add_argument("-k", type=int, default=5, help="Number of retrieved chunks per query (default: 5)")
    parser.add_argument("--persona", type=str, default="assistant", choices=["detective", "assistant", "strict_analyst", "academic"], help="System persona")

    args = parser.parse_args()

    print(BANNER)

    rag = RAGPipeline(
        llm_model=args.llm,
        embed_model=args.embed,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        k=args.k,
        persona=args.persona
    )

    last_sources = []

    # Determine initial data source
    if args.url:
        print(f"Fetching and indexing web link: {args.url}...")
        try:
            summary_res = rag.summarize_url(args.url)
            last_sources = summary_res["sources"]
            print(f"Success! Indexed web page: \"{summary_res['title']}\" into {summary_res['stats']['total_chunks']} chunks.\n")
            print("=" * 80)
            print(summary_res["summary"])
            print("=" * 80)
            if summary_res["suggested_questions"]:
                print("\n💡 Suggested Exploration Questions:")
                for idx, q in enumerate(summary_res["suggested_questions"], 1):
                    print(f"   [{idx}] {q}")
            print("\nYou can now ask any questions about this web page!\n")
        except Exception as e:
            print(f"Error processing URL: {e}")
            sys.exit(1)
    elif args.load_index:
        print(f"Loading existing vector database from: {args.load_index}...")
        try:
            rag.load_index(args.load_index)
            print("Vector database loaded successfully!\n")
        except Exception as e:
            print(f"Error loading index: {e}")
            sys.exit(1)
    elif args.dir or args.files:
        targets = args.files if args.files else args.dir
        print(f"Indexing documents from: {targets}...")
        try:
            stats = rag.build_index_from_files(targets)
            print(f"Success! Indexed {stats['files_count']} file(s), {stats['total_pages']} page(s) into {stats['total_chunks']} chunks.\n")
        except Exception as e:
            print(f"Error indexing documents: {e}")
            sys.exit(1)
    else:
        if os.path.exists("elrond_investigation"):
            print("Found existing 'elrond_investigation' vector database. Loading it...")
            rag.load_index("elrond_investigation")
            print("Vector database loaded successfully!\n")
        elif os.path.exists("data"):
            print("Found 'data' folder. Indexing documents...")
            stats = rag.build_index_from_files("data")
            print(f"Success! Indexed {stats['files_count']} file(s) into {stats['total_chunks']} chunks.\n")
        else:
            path_input = input("Enter path to documents folder, file, or URL to index: ").strip()
            if path_input.startswith("http://") or path_input.startswith("https://"):
                stats = rag.build_index_from_url(path_input)
                print(f"Success! Indexed web page into {stats['total_chunks']} chunks.\n")
            elif path_input:
                stats = rag.build_index_from_files(path_input)
                print(f"Success! Indexed {stats['files_count']} file(s) into {stats['total_chunks']} chunks.\n")
            else:
                print("No documents provided. Exiting.")
                sys.exit(0)

    print("System ready! Type your question, '/url <link>' to ingest a web page, or '/help' for commands.\n")

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd == "/help":
                print(HELP_TEXT)
            elif cmd == "/url":
                if not arg:
                    print("Usage: /url <https://example.com/article>")
                else:
                    print(f"\nFetching and indexing: {arg}...", end="\r", flush=True)
                    try:
                        summary_res = rag.summarize_url(arg)
                        last_sources = summary_res["sources"]
                        print(" " * 60, end="\r")
                        print("\n" + "=" * 80)
                        print(f"SOURCE: {summary_res['title']} ({summary_res['url']})")
                        print("=" * 80)
                        print(summary_res["summary"])
                        print("=" * 80)
                        if summary_res["suggested_questions"]:
                            print("\n💡 Suggested Questions:")
                            for idx, q in enumerate(summary_res["suggested_questions"], 1):
                                print(f"   [{idx}] {q}")
                        print("\nYou can now ask any question about this web page!")
                    except Exception as e:
                        print(f"\nError ingesting URL: {e}")
            elif cmd == "/summary":
                print("\nSynthesizing Executive Briefing across documents...", end="\r", flush=True)
                try:
                    summary_data = rag.generate_summary()
                    last_sources = summary_data["sources"]
                    print(" " * 55, end="\r")
                    print("\n" + "=" * 80)
                    print(summary_data["summary"])
                    print("=" * 80)
                    if summary_data["suggested_questions"]:
                        print("\n💡 Suggested Exploration Questions:")
                        for idx, q in enumerate(summary_data["suggested_questions"], 1):
                            print(f"   [{idx}] {q}")
                    print("\n[Type any of the questions above or enter a new question]")
                except Exception as e:
                    print(f"\nError generating summary: {e}")
            elif cmd == "/sources":
                print(format_sources(last_sources))
            elif cmd == "/stats":
                stats = rag.get_stats()
                print("\n" + "-" * 40)
                for k, v in stats.items():
                    print(f"  {k}: {v}")
                print("-" * 40 + "\n")
            elif cmd == "/clear":
                rag.clear_chat_history()
                print("Conversation history cleared.")
            elif cmd == "/persona":
                if arg in rag.PERSONAS:
                    rag.update_hyperparameters(persona=arg)
                    print(f"Persona switched to '{arg}'.")
                else:
                    print(f"Unknown persona. Choose from: {list(rag.PERSONAS.keys())}")
            elif cmd == "/save":
                if not arg:
                    print("Usage: /save <directory_name>")
                else:
                    rag.save_index(arg)
                    print(f"Index successfully saved to '{arg}'.")
            elif cmd == "/load":
                if not arg:
                    print("Usage: /load <directory_name>")
                else:
                    try:
                        rag.load_index(arg)
                        print(f"Index successfully loaded from '{arg}'.")
                    except Exception as e:
                        print(f"Failed to load: {e}")
            elif cmd == "/index":
                if not arg:
                    print("Usage: /index <file_or_folder_path>")
                else:
                    try:
                        s = rag.build_index_from_files(arg)
                        print(f"Indexed {s['files_count']} file(s) into {s['total_chunks']} chunks.")
                    except Exception as e:
                        print(f"Failed to index: {e}")
            else:
                print(f"Unknown command '{cmd}'. Type /help for options.")
            continue

        # Normal question
        try:
            print("\nAI is thinking...", end="\r", flush=True)
            result = rag.ask(user_input)
            last_sources = result["sources"]

            print(" " * 30, end="\r")  # Clear thinking message
            print(f"Assistant ({rag.llm_model_name}):")
            print("-" * 80)
            print(result["answer"])
            print("-" * 80)
            avg_rel = ""
            if last_sources:
                rels = [s.get("relevance_num", 0) for s in last_sources if "relevance_num" in s]
                if rels:
                    avg_rel = f" | Avg match: {round(sum(rels)/len(rels), 1)}%"
            print(f"[Retrieved {len(last_sources)} chunks{avg_rel}. Type '/sources' to inspect chunks]")

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
