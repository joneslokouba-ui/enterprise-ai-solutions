import os
import sys
import logging

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import streamlit as st
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT_DIR, ".env"))

from src.embedder import TaxEmbedder
from src.retriever import TaxRetriever

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Tax RAG Assistant",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_rag_system():
    """
    Load embedder and retriever once.
    @st.cache_resource means this runs only on first load.
    """
    with st.spinner("Loading Tax RAG Assistant..."):
        embedder = TaxEmbedder()
        if embedder.index_exists():
            vectorstore = embedder.load_index()
        else:
            st.info("Building knowledge base for first time...")
            from src.loader import TaxDocumentLoader
            docs_folder = os.getenv("DOCS_FOLDER", "data")
            loader = TaxDocumentLoader(
                docs_folder=docs_folder,
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = loader.load()
            vectorstore = embedder.build_index(chunks)

        retriever = TaxRetriever(vectorstore)
        return retriever


def render_sidebar():
    """Render sidebar with app info."""
    with st.sidebar:
        st.title("🦅 Tax RAG Assistant")
        st.caption("v1.0.0 — Built by Geoffrey Jones Okwi")
        st.divider()

        st.subheader("📚 Knowledge Base")
        st.write("**Documents loaded:**")
        st.write("- IRS Publication 17 (p17.pdf)")
        st.write("- IRS Form 1040 Instructions")
        st.write("- 268 pages | 3,797 chunks")
        st.divider()

        st.subheader("⚙️ Settings")
        st.write(f"**Model:** llama-3.3-70b-versatile")
        st.write(f"**Embeddings:** all-MiniLM-L6-v2")
        st.write(f"**Vector store:** FAISS")
        st.divider()

        st.subheader("💡 Sample questions")
        questions = [
            "What is the standard deduction?",
            "How do I report freelance income?",
            "What are the tax brackets?",
            "How do I claim the child tax credit?",
            "What is a W-2 form?"
        ]
        for q in questions:
            if st.button(q, use_container_width=True):
                st.session_state.sample_question = q


def render_chat(retriever):
    """Render the main chat interface."""
    st.title("🦅 Tax RAG Assistant")
    st.caption(
        "Ask any US federal tax question — answers cited "
        "from IRS documents."
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Handle sample question clicks from sidebar
    if "sample_question" in st.session_state:
        question = st.session_state.pop("sample_question")
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        with st.spinner("Searching tax documents..."):
            response = retriever.ask(question)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                response = message["content"]
                if response.get("error"):
                    st.error(f"Error: {response['error']}")
                else:
                    st.markdown(response["answer"])
                    if response["sources"]:
                        with st.expander(
                            f"📄 Sources ({len(response['sources'])})"
                        ):
                            for source in response["sources"]:
                                st.markdown(
                                    f"**{source['filename']}** — "
                                    f"Page {source['page']}"
                                )
                                st.caption(source["preview"])

    # Chat input
    if question := st.chat_input(
        "Ask a tax question..."
    ):
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching IRS documents..."):
                response = retriever.ask(question)

            if response.get("error"):
                st.error(f"Error: {response['error']}")
            else:
                st.markdown(response["answer"])
                if response["sources"]:
                    with st.expander(
                        f"📄 Sources ({len(response['sources'])})"
                    ):
                        for source in response["sources"]:
                            st.markdown(
                                f"**{source['filename']}** — "
                                f"Page {source['page']}"
                            )
                            st.caption(source["preview"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })


def main():
    """Main entry point."""
    render_sidebar()

    try:
        retriever = load_rag_system()
        render_chat(retriever)
    except Exception as e:
        st.error(f"Failed to load RAG system: {e}")
        st.info("Check your .env file and USB drive connection.")


if __name__ == "__main__":
    main()