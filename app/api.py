# Fix OpenMP library conflict and suppress warnings
import os
import sys
import warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore', category=UserWarning)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import shutil
from app.config import (
    VECTOR_STORE_PATH,
    EMBEDDING_PROVIDER,
    LLM_PROVIDER,
    VECTOR_STORE_PROVIDER,
    VECTOR_STORE_CONFIG,
    CATEGORY_KEYWORDS,
    PINECONE_API_KEY,
)
from app.ingestion import scan_knowledge_base, chunk_documents
from app.embedding import create_vector_store, save_vector_store, load_vector_store, save_file_metadata
from app.retrieval import retrieve_documents
from app.reranker import rerank_documents
from app.generation import generate_answer

# Page config
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📘",
    layout="wide"
)

# Header
st.markdown("""
# 📘 KT Onboarding Assistant
Welcome!  
Browse documents from the **knowledge-base** folder organized by product/category.

This assistant answers **only from your knowledge base**.
""")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}

# Build knowledge base
def build_knowledge_base():
    with st.spinner("Building knowledge base..."):
        categories = scan_knowledge_base()
        if not categories:
            st.error("No documents found")
            return
        
        chunks, metadatas, links, file_metadata = chunk_documents(categories)
        
        if chunks:
            vectorstore = create_vector_store(chunks, metadatas)
            save_vector_store(vectorstore)
            save_file_metadata(file_metadata)
            st.session_state.vectorstore = vectorstore
            st.success(f"✅ Knowledge base built with {len(chunks)} chunks from {len(categories)} categories")
        else:
            st.warning("No content found to process")

# Load existing vector store
if st.session_state.vectorstore is None:
    st.session_state.vectorstore = load_vector_store()
    if st.session_state.vectorstore:
        st.info("✅ Loaded existing knowledge base")

# Sidebar
st.sidebar.markdown("### 📊 Active Providers")
st.sidebar.markdown(f"**Embedding:** {EMBEDDING_PROVIDER}")
st.sidebar.markdown(f"**LLM:** {LLM_PROVIDER}")
st.sidebar.markdown(f"**Vector Store:** {VECTOR_STORE_PROVIDER}")
st.sidebar.markdown("---")

if st.sidebar.button("🔄 Rebuild Knowledge Base"):
    build_knowledge_base()

if st.sidebar.button("🗑️ Clear Knowledge Base"):
    if VECTOR_STORE_PROVIDER == "pinecone":
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(VECTOR_STORE_CONFIG["index_name"])
            index.delete(delete_all=True, namespace=VECTOR_STORE_CONFIG.get("namespace"))
        except Exception as e:
            st.sidebar.error(f"Failed to clear Pinecone index: {e}")
    else:
        if os.path.exists(VECTOR_STORE_PATH):
            shutil.rmtree(VECTOR_STORE_PATH)
    st.session_state.vectorstore = None
    st.session_state.messages = []
    st.session_state.response_cache = {}
    st.sidebar.success("Knowledge base cleared")
    st.rerun()

# Process question
def process_question(question):
    if st.session_state.vectorstore is not None:
        question_key = question.lower().strip()
        
        if question_key in st.session_state.response_cache:
            cached_response = st.session_state.response_cache[question_key]
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": cached_response})
            return
        
        st.session_state.messages.append({"role": "user", "content": question})
        
        # RAG Pipeline
        docs = retrieve_documents(st.session_state.vectorstore, question)
        reranked_docs = rerank_documents(question, docs)
        answer = generate_answer(question, reranked_docs)
        
        st.session_state.response_cache[question_key] = answer
        st.session_state.messages.append({"role": "assistant", "content": answer})

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if st.session_state.vectorstore is not None:
    question = st.chat_input("Ask a question about your documents...")
    if question:
        process_question(question)
        st.rerun()
else:
    st.warning("⚠️ No knowledge base loaded. Click 'Rebuild Knowledge Base' in the sidebar.")

# Footer
st.markdown("---")
st.caption("🔒 Document Q&A Assistant | Answers strictly from knowledge base")
