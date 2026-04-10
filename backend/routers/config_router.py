"""
Config router — provider configuration endpoint.

Requirements: 7.1, 7.2
"""

from fastapi import APIRouter, Depends

from app.config import EMBEDDING_PROVIDER, LLM_PROVIDER, VECTOR_STORE_PROVIDER
from backend.dependencies import require_admin
from backend.models import ProviderConfigResponse

router = APIRouter(prefix="/api", tags=["config"])


@router.get("/config", response_model=ProviderConfigResponse)
async def get_config(
    _admin: dict = Depends(require_admin),
) -> ProviderConfigResponse:
    """Return the active embedding, LLM, and vector store provider names."""
    return ProviderConfigResponse(
        embedding_provider=EMBEDDING_PROVIDER,
        llm_provider=LLM_PROVIDER,
        vector_store_provider=VECTOR_STORE_PROVIDER,
    )
