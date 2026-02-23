# RAG Classic - Chatbot

A modular Classic RAG implementation for document Q&A using AWS Bedrock.

## Architecture

```
rag-classic/
├── app/
│   ├── __init__.py        # Package marker
│   ├── config.py          # Settings & environment variables
│   ├── ingestion.py       # Document loading & chunking
│   ├── embedding.py       # FAISS vector store operations
│   ├── retrieval.py       # Semantic search
│   ├── reranker.py        # Embedding-based reranking
│   ├── generation.py      # LLM answer generation
│   └── api.py             # Streamlit UI
├── knowledge-base/        # Source documents (organized by category)
├── main.py                # Entry point
└── README.md
```

## Features

✅ **Document Ingestion** - PDF, DOCX, HTML, TXT, MD support  
✅ **Embeddings** - AWS Bedrock Titan embeddings  
✅ **Vector Store** - FAISS for efficient similarity search  
✅ **Retrieval** - Category-aware semantic search  
✅ **Reranking** - Embedding similarity scoring  
✅ **Generation** - Claude 3 Haiku with strict grounding  

## Setup

1. Ensure AWS credentials are configured
2. Place documents in `knowledge-base/` folder organized by category
3. Install dependencies:
   ```bash
   pip install streamlit boto3 PyPDF2 python-docx beautifulsoup4 langchain langchain-aws langchain-community faiss-cpu numpy
   ```

## Usage

Run the application:
```bash
streamlit run main.py
```

Or run the original file:
```bash
streamlit run app/api.py
```

## RAG Pipeline

1. **Ingestion** - Extract text from documents and chunk
2. **Embedding** - Create embeddings and store in FAISS
3. **Retrieval** - Search top-k similar chunks (k=10)
4. **Reranking** - Rerank using embedding similarity (top-3)
5. **Generation** - Generate grounded answer with citations
