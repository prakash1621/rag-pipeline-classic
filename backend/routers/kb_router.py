"""
Knowledge-base operations router — rebuild and clear the vector store.

Requirements: 6.1, 6.2, 6.3, 6.4
"""

from fastapi import APIRouter, Depends

from app.embedding import create_vector_store, save_vector_store
from app.ingestion import scan_knowledge_base, chunk_documents
from backend import state
from backend.dependencies import require_admin
from backend.models import RebuildResponse

router = APIRouter(prefix="/api/knowledge-base", tags=["knowledge-base"])


@router.post("/rebuild", response_model=RebuildResponse)
async def rebuild_knowledge_base(_user: dict = Depends(require_admin)):
    """Scan documents, chunk, embed, and persist the vector store."""
    with state.vectorstore_lock:
        categories = scan_knowledge_base()
        chunks, metadatas, _links, _file_metadata = chunk_documents(categories)
        vectorstore = create_vector_store(chunks, metadatas)
        save_vector_store(vectorstore)
        state.vectorstore = vectorstore

    return RebuildResponse(chunks=len(chunks), categories=len(categories))


@router.post("/clear")
async def clear_knowledge_base(_user: dict = Depends(require_admin)):
    """Reset the in-memory vector store to None."""
    with state.vectorstore_lock:
        state.vectorstore = None

    return {"detail": "Knowledge base cleared"}
