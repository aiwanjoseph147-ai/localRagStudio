#!/usr/bin/env python3
"""
RAG Studio Web Application
Local HTTP Server serving a modern interactive UI for Document RAG with Ollama & FAISS.
Features:
- File uploads (PDF, TXT, MD, CSV)
- Web URL fetching, indexing, and instant summarization
- Grounded Q&A with conversational memory
- Executive Document Briefing generator
- Vector relevance match percentage scoring
"""

import os
import sys
import json
import base64
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

from rag_pipeline import RAGPipeline, list_ollama_models

PORT = 8000
HOST = "127.0.0.1"

# Initialize global RAG instance
rag = RAGPipeline()

# Auto-load existing index if available
if os.path.exists("elrond_investigation"):
    try:
        print("Auto-loading existing 'elrond_investigation' FAISS index...")
        rag.load_index("elrond_investigation")
        print("Index loaded successfully!")
    except Exception as e:
        print(f"Notice: Could not auto-load elrond_investigation: {e}")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    daemon_threads = True


class RAGRequestHandler(SimpleHTTPRequestHandler):
    """Custom request handler providing REST API and serving UI."""

    def end_headers(self):
        # Enable CORS for local development
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")
        if not path:
            path = "/"

        if path in ("/", "/index.html"):
            self.serve_file(os.path.join(os.path.dirname(__file__), "web", "index.html"), "text/html")
        elif path == "/api/status":
            self.send_json(rag.get_stats())
        elif path == "/api/models":
            models = list_ollama_models()
            self.send_json({"models": models})
        elif path.startswith("/api/"):
            self.send_json({"error": f"API GET endpoint not found: {path}"}, status=404)
        else:
            # Fallback for static assets in web/
            local_path = os.path.join(os.path.dirname(__file__), "web", path.lstrip("/"))
            if os.path.exists(local_path) and os.path.isfile(local_path):
                self.serve_file(local_path)
            else:
                self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/")

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception as e:
            self.send_json({"error": f"Invalid JSON payload: {e}"}, status=400)
            return

        try:
            if path == "/api/chat":
                self.handle_chat(body)
            elif path == "/api/summary":
                self.handle_summary(body)
            elif path == "/api/url-summary":
                self.handle_url_summary(body)
            elif path == "/api/upload":
                self.handle_upload(body)
            elif path == "/api/load-directory":
                self.handle_load_directory(body)
            elif path == "/api/load-saved-index":
                self.handle_load_saved_index(body)
            elif path == "/api/save-index":
                self.handle_save_index(body)
            elif path == "/api/clear":
                rag.clear_chat_history()
                self.send_json({"status": "cleared"})
            else:
                self.send_json({"error": f"API POST endpoint not found: {path}"}, status=404)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json({"error": str(e)}, status=500)

    def handle_chat(self, body):
        question = body.get("question", "").strip()
        if not question:
            self.send_json({"error": "No question provided"}, status=400)
            return

        k = body.get("k", 5)
        persona = body.get("persona", "assistant")
        llm_model = body.get("llm_model", "qwen2.5:3b")
        history = body.get("history", [])

        rag.update_models(llm_model=llm_model)
        rag.update_hyperparameters(k=k, persona=persona)

        result = rag.ask(question, chat_history=history, k=k)
        self.send_json(result)

    def handle_summary(self, body):
        max_chunks = body.get("max_chunks", 6)
        llm_model = body.get("llm_model")
        if llm_model:
            rag.update_models(llm_model=llm_model)
        result = rag.generate_summary(max_chunks=max_chunks)
        self.send_json(result)

    def handle_url_summary(self, body):
        url = body.get("url", "").strip()
        if not url:
            self.send_json({"error": "No URL provided"}, status=400)
            return

        chunk_size = body.get("chunk_size", 500)
        chunk_overlap = body.get("chunk_overlap", 150)
        llm_model = body.get("llm_model")
        embed_model = body.get("embed_model")
        max_chunks = body.get("max_chunks", 6)

        if llm_model or embed_model:
            rag.update_models(llm_model=llm_model, embed_model=embed_model)

        result = rag.summarize_url(
            url,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks
        )
        self.send_json(result)

    def handle_upload(self, body):
        files_data = body.get("files", [])
        if not files_data:
            self.send_json({"error": "No files received"}, status=400)
            return

        upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        for f in os.listdir(upload_dir):
            try:
                os.remove(os.path.join(upload_dir, f))
            except Exception:
                pass

        saved_paths = []
        for file_item in files_data:
            name = os.path.basename(file_item.get("name", "document.txt"))
            data_str = file_item.get("data", "")
            if "," in data_str:
                data_str = data_str.split(",", 1)[1]

            file_bytes = base64.b64decode(data_str)
            target_path = os.path.join(upload_dir, name)
            with open(target_path, "wb") as f:
                f.write(file_bytes)
            saved_paths.append(target_path)

        chunk_size = body.get("chunk_size", 500)
        chunk_overlap = body.get("chunk_overlap", 150)
        llm_model = body.get("llm_model")
        embed_model = body.get("embed_model")

        rag.update_models(llm_model=llm_model, embed_model=embed_model)
        stats = rag.build_index_from_files(
            saved_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.send_json(stats)

    def handle_load_directory(self, body):
        directory = body.get("directory", "data")
        chunk_size = body.get("chunk_size", 500)
        chunk_overlap = body.get("chunk_overlap", 150)
        llm_model = body.get("llm_model")
        embed_model = body.get("embed_model")

        rag.update_models(llm_model=llm_model, embed_model=embed_model)
        stats = rag.build_index_from_files(
            directory,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.send_json(stats)

    def handle_load_saved_index(self, body):
        index_path = body.get("index_path", "elrond_investigation")
        stats = rag.load_index(index_path)
        self.send_json(stats)

    def handle_save_index(self, body):
        index_path = body.get("index_path", "saved_index")
        saved_to = rag.save_index(index_path)
        self.send_json({"status": "success", "index_path": saved_to})

    def send_json(self, data, status=200):
        json_bytes = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(json_bytes)))
        self.end_headers()
        self.wfile.write(json_bytes)

    def serve_file(self, filepath, mime_type=None):
        if not os.path.exists(filepath):
            self.send_error(404, "File not found")
            return

        with open(filepath, "rb") as f:
            content = f.read()

        if not mime_type:
            ext = os.path.splitext(filepath)[1].lower()
            mime_map = {
                ".html": "text/html",
                ".css": "text/css",
                ".js": "application/javascript",
                ".json": "application/json",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".svg": "image/svg+xml"
            }
            mime_type = mime_map.get(ext, "application/octet-stream")

        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def run_server(port=PORT):
    server_address = (HOST, port)
    server = ThreadedHTTPServer(server_address, RAGRequestHandler)
    print(f"\n========================================================")
    print(f" 🚀 RAG Studio Web App is running!")
    print(f" 👉 Open in your browser: http://{HOST}:{port}")
    print(f"========================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()


if __name__ == "__main__":
    port_to_use = PORT
    if len(sys.argv) > 1:
        try:
            port_to_use = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port_to_use)
