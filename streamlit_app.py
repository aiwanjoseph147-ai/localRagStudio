"""
Streamlit Web UI for RAG Pipeline with Ollama & FAISS
Features grounded Q&A, Executive Briefing generator, and relevance scoring.
Launch with: streamlit run streamlit_app.py
"""

import os
import tempfile
import streamlit as st
from rag_pipeline import RAGPipeline, list_ollama_models

st.set_page_config(
    page_title="RAG Document QA Studio",
    page_icon="🔍",
    layout="wide"
)

# Initialize Session State
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()
    if os.path.exists("elrond_investigation"):
        try:
            st.session_state.rag.load_index("elrond_investigation")
        except Exception:
            pass

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ RAG Configuration")

    st.subheader("1. Document Management")
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Index Uploaded Files", disabled=not uploaded_files, use_container_width=True):
            with st.spinner("Processing documents & computing embeddings..."):
                temp_dir = tempfile.mkdtemp()
                paths = []
                for uf in uploaded_files:
                    target = os.path.join(temp_dir, uf.name)
                    with open(target, "wb") as f:
                        f.write(uf.getbuffer())
                    paths.append(target)
                
                stats = st.session_state.rag.build_index_from_files(paths)
                st.success(f"Indexed {stats['files_count']} file(s) into {stats['total_chunks']} chunks!")

    with col2:
        if st.button("Load ./data Folder", use_container_width=True):
            if os.path.exists("data"):
                with st.spinner("Indexing ./data PDFs..."):
                    stats = st.session_state.rag.build_index_from_files("data")
                    st.success(f"Loaded {stats['files_count']} files into {stats['total_chunks']} chunks!")
            else:
                st.error("./data folder not found")

    st.divider()

    st.subheader("2. Knowledge Base & Analysis")
    stats = st.session_state.rag.get_stats()
    if stats["has_index"]:
        st.success("🟢 Index is active")
        st.write(f"**Chunks:** {stats['total_chunks']} | **Files:** {stats['total_files']}")
    else:
        st.warning("🟡 No documents loaded yet")

    # NEW FEATURE: Executive Briefing Button
    if st.button("📑 Generate Executive Briefing", use_container_width=True, disabled=not stats["has_index"]):
        with st.spinner("Synthesizing Executive Briefing across documents..."):
            try:
                briefing_data = st.session_state.rag.generate_summary()
                st.session_state.messages.append({
                    "role": "user",
                    "content": "Generate an Executive Briefing of the documents."
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": briefing_data["summary"],
                    "sources": briefing_data["sources"],
                    "suggested_questions": briefing_data["suggested_questions"]
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error generating briefing: {e}")

    st.divider()

    st.subheader("3. Hyperparameters")
    models = list_ollama_models()
    selected_llm = st.selectbox("LLM Model", options=models, index=0 if "qwen2.5:3b" not in models else models.index("qwen2.5:3b"))
    selected_embed = st.selectbox("Embedding Model", options=["bge-m3"], index=0)

    chunk_size = st.slider("Chunk Size (characters)", 200, 1500, 500, step=50)
    chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 400, 150, step=25)
    k_val = st.slider("Top-K Retrieved Chunks", 1, 15, 5)

    persona = st.selectbox(
        "AI Persona / Prompt",
        options=["detective", "assistant", "strict_analyst", "academic"],
        format_func=lambda x: {
            "detective": "AI Detective (Case Investigation)",
            "assistant": "Helpful Assistant",
            "strict_analyst": "Strict Fact-Checker",
            "academic": "Academic Researcher"
        }.get(x, x)
    )

    st.session_state.rag.update_models(llm_model=selected_llm, embed_model=selected_embed)
    st.session_state.rag.update_hyperparameters(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        k=k_val,
        persona=persona
    )

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.rag.clear_chat_history()
        st.rerun()

# Main Area
st.title("🔍 RAG Document QA Studio")
st.caption("Ask questions grounded strictly in your documents with Ollama & FAISS")

# Display Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display suggested questions if available
        if "suggested_questions" in msg and msg["suggested_questions"]:
            st.markdown("##### 💡 Recommended Exploration Questions:")
            for q in msg["suggested_questions"]:
                st.markdown(f"- *{q}*")

        # Display sources with relevance match %
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📚 Retrieved Sources ({len(msg['sources'])} chunks)"):
                for s in msg["sources"]:
                    rel_text = f" • Match: {s['relevance_score']}" if "relevance_score" in s else ""
                    st.markdown(f"**📄 {s['file_name']} (Page {s['page']}){rel_text}**")
                    st.text(s["content"])
                    st.divider()

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            try:
                res = st.session_state.rag.ask(prompt)
                st.markdown(res["answer"])

                if res["sources"]:
                    with st.expander(f"📚 Retrieved Sources ({len(res['sources'])} chunks)"):
                        for s in res["sources"]:
                            rel_text = f" • Match: {s['relevance_score']}" if "relevance_score" in s else ""
                            st.markdown(f"**📄 {s['file_name']} (Page {s['page']}){rel_text}**")
                            st.text(s["content"])
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": res["answer"],
                    "sources": res["sources"]
                })
            except Exception as e:
                st.error(f"Error generating answer: {e}")
