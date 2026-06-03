import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.workflow import rag_app
from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from retrieval.vectorstore import store_documents

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Adaptive Research Assistant")
st.caption("Powered by CRAG · Hybrid Search · LangGraph · Groq")

# ── Sidebar — Document Ingestion ───────────────────────────
with st.sidebar:
    st.header("📚 Knowledge Base")

    st.subheader("Add URLs")
    url_input = st.text_area(
        "Enter URLs (one per line)",
        placeholder="https://en.wikipedia.org/wiki/..."
    )

    st.subheader("Add PDFs")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("📥 Ingest Documents", use_container_width=True):
        sources = []

        # Collect URLs
        if url_input.strip():
            sources.extend([
                url.strip()
                for url in url_input.strip().split("\n")
                if url.strip()
            ])

        # Save and collect PDFs
        if uploaded_files:
            os.makedirs("./temp_pdfs", exist_ok=True)
            for f in uploaded_files:
                path = f"./temp_pdfs/{f.name}"
                with open(path, "wb") as out:
                    out.write(f.read())
                sources.append(path)

        if sources:
            with st.spinner("Ingesting documents..."):
                docs = load_documents(sources)
                chunks = chunk_documents(docs)
                store_documents(chunks)
            st.success(f"✅ Ingested {len(chunks)} chunks from {len(sources)} sources!")
        else:
            st.warning("Please add at least one URL or PDF")

    st.divider()
    st.caption("Built with LangGraph + Groq + Qdrant")

# ── Main — Chat Interface ───────────────────────────────────

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("scores"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Faithfulness", msg["scores"].get("faithfulness", "N/A"))
            col2.metric("Relevancy", msg["scores"].get("answer_relevancy", "N/A"))
            col3.metric("Context Recall", msg["scores"].get("context_recall", "N/A"))

# Chat input
if query := st.chat_input("Ask a research question..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):

            initial_state = {
                "query": query,
                "enhanced_queries": None,
                "retrieved_docs": None,
                "grade": None,
                "rewrite_count": 0,
                "answer": None,
                "eval_scores": None
            }

            final_state = rag_app.invoke(initial_state)
            answer = final_state["answer"]
            scores = final_state.get("eval_scores")
            grade = final_state.get("grade")

        st.markdown(answer)

        # Show metadata
        col1, col2 = st.columns(2)
        col1.info(f"Grade: {grade}")
        col2.info(f"Rewrites: {final_state['rewrite_count']}")

        # Show RAGAS scores if available
        if scores:
            st.subheader("📊 Answer Quality")
            c1, c2, c3 = st.columns(3)
            c1.metric("Faithfulness", scores.get("faithfulness", "N/A"))
            c2.metric("Relevancy", scores.get("answer_relevancy", "N/A"))
            c3.metric("Context Recall", scores.get("context_recall", "N/A"))

    # Store in history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "scores": scores
    })