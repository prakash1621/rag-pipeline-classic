"""
Query router — POST /api/query endpoint.

Passes the user's question through the RAG pipeline (retrieval, reranking,
generation) and returns the answer with citations.

Requirements: 4.1, 4.2, 4.3, 4.4
"""

from fastapi import APIRouter, Depends, HTTPException, status

from backend import state
from backend.dependencies import get_current_user
from backend.models import QueryRequest, QueryResponse
from app.retrieval import retrieve_documents
from app.reranker import rerank_documents
from app.generation import generate_answer

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest, user: dict = Depends(get_current_user)):
    """Accept a question, run the RAG pipeline, return answer + citations."""
    if state.vectorstore is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base is not loaded. Rebuild required.",
        )

    docs = retrieve_documents(state.vectorstore, body.question)
    reranked = rerank_documents(body.question, docs)
    result = generate_answer(body.question, reranked)

    # generate_answer returns answer text followed by "\n\n---\n" and citations
    if "---" in result:
        parts = result.split("---", 1)
        answer = parts[0].strip()
        citations = parts[1].strip()
    else:
        answer = result.strip()
        citations = ""

    return QueryResponse(answer=answer, citations=citations)
