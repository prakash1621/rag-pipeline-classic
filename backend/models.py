from pydantic import BaseModel, Field
from typing import Literal


# ── Auth ────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    role: Literal["admin", "viewer"]


class RefreshRequest(BaseModel):
    refresh_token: str


class AccessTokenResponse(BaseModel):
    access_token: str


# ── Query ───────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    answer: str
    citations: str


# ── Users ───────────────────────────────────────────────────
class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=6)
    role: Literal["admin", "viewer"]


class UserOut(BaseModel):
    username: str
    role: Literal["admin", "viewer"]


# ── Documents ───────────────────────────────────────────────
class DocumentListResponse(BaseModel):
    documents: dict[str, list[str]]  # category -> [filenames]


# ── Knowledge Base ──────────────────────────────────────────
class RebuildResponse(BaseModel):
    chunks: int
    categories: int


# ── Config ──────────────────────────────────────────────────
class ProviderConfigResponse(BaseModel):
    embedding_provider: str
    llm_provider: str
    vector_store_provider: str
