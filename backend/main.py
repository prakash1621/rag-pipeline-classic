"""
FastAPI application entry point.

Run with:  uvicorn backend.main:app --reload
from the rag-pipeline-classic/ directory.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routers.auth_router import router as auth_router
from backend.routers.query_router import router as query_router
from backend.routers.users_router import router as users_router
from backend.routers.documents_router import router as documents_router
from backend.routers.kb_router import router as kb_router
from backend.routers.config_router import router as config_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create default admin and attempt to load the vectorstore."""
    from backend.auth import ensure_default_admin
    from backend import state

    # Ensure a default admin account exists
    ensure_default_admin()

    # Attempt to load an existing vectorstore into memory
    try:
        from app.embedding import load_vector_store

        vs = load_vector_store()
        if vs is not None:
            state.vectorstore = vs
            logger.info("Vectorstore loaded successfully on startup.")
        else:
            logger.info("No existing vectorstore found. Rebuild required.")
    except Exception as exc:
        logger.warning("Could not load vectorstore on startup: %s", exc)

    yield


app = FastAPI(title="RAG Pipeline API", lifespan=lifespan)

# CORS — allow all origins, credentials, methods, and headers (Req 14.1, 14.2)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth_router)
app.include_router(query_router)
app.include_router(users_router)
app.include_router(documents_router)
app.include_router(kb_router)
app.include_router(config_router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions — return 500 with no stack trace."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
