"""
Config_Loader — loads all settings from config.yaml and .env files.

All other modules import constants from here. No module should read
YAML or .env directly.
"""

import os

import yaml
from dotenv import load_dotenv

# ── Base directories ────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))        # rag-pipeline-classic/
WORKSPACE_ROOT = os.path.dirname(BASE_DIR)                   # workspace root


def load_config(yaml_path: str | None = None) -> dict:
    """Load config.yaml and merge .env overrides. Returns the raw dict.

    Raises FileNotFoundError with a descriptive message when config.yaml
    is missing or unreadable.
    """
    if yaml_path is None:
        # Look in repo root first (for Streamlit Cloud), then workspace root (local dev)
        repo_path = os.path.join(BASE_DIR, "config.yaml")
        workspace_path = os.path.join(WORKSPACE_ROOT, "config.yaml")
        if os.path.isfile(repo_path):
            yaml_path = repo_path
        else:
            yaml_path = workspace_path

    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(
            f"Configuration file not found: {yaml_path}. "
            "Please create a config.yaml at the workspace root."
        )

    with open(yaml_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Load .env — check repo root first, then workspace root
    repo_env = os.path.join(BASE_DIR, ".env")
    workspace_env = os.path.join(WORKSPACE_ROOT, ".env")
    env_path = repo_env if os.path.isfile(repo_env) else workspace_env
    load_dotenv(env_path)

    # Support Streamlit Cloud secrets (st.secrets → env vars)
    try:
        import streamlit as st
        for key in ["PINECONE_API_KEY", "OPENAI_API_KEY", "TAVILY_API_KEY",
                     "AWS_DEFAULT_REGION", "AWS_EMBEDDING_MODEL", "AWS_LLM_MODEL",
                     "SEC_USER_AGENT"]:
            if key in st.secrets and key not in os.environ:
                os.environ[key] = st.secrets[key]
    except Exception:
        pass

    # .env overrides for paths
    env_vs = os.environ.get("VECTOR_STORE_PATH")
    env_kb = os.environ.get("KNOWLEDGE_BASE_PATH")
    env_cache = os.environ.get("CACHE_DIR")

    paths = cfg.get("paths", {})
    if env_vs:
        paths["vector_store"] = env_vs
    if env_kb:
        paths["knowledge_base"] = env_kb
    if env_cache:
        paths["cache_dir"] = env_cache
    cfg["paths"] = paths

    return cfg


# ── Load once at import time ────────────────────────────────
_cfg = load_config()

# ── Embedding ───────────────────────────────────────────────
EMBEDDING_PROVIDER: str = _cfg["embedding"]["provider"]
EMBEDDING_CONFIG: dict = _cfg["embedding"].get(EMBEDDING_PROVIDER, {})

# ── LLM ─────────────────────────────────────────────────────
LLM_PROVIDER: str = _cfg["llm"]["provider"]
LLM_CONFIG: dict = _cfg["llm"].get(LLM_PROVIDER, {})

# ── Vector Store ────────────────────────────────────────────
VECTOR_STORE_PROVIDER: str = _cfg["vector_store"]["provider"]
VECTOR_STORE_CONFIG: dict = _cfg["vector_store"].get(VECTOR_STORE_PROVIDER, {})

# ── Chunking ────────────────────────────────────────────────
CHUNKING_CONFIG: dict = _cfg.get("chunking", {})

# ── Retrieval ───────────────────────────────────────────────
RETRIEVAL_K: int = _cfg["retrieval"]["k"]
RERANK_TOP_K: int = _cfg["retrieval"]["rerank_top_k"]

# ── Categories ──────────────────────────────────────────────
CATEGORY_KEYWORDS: dict = _cfg.get("categories", {})

# ── Paths (with .env overrides already applied) ─────────────
PATHS_CONFIG: dict = _cfg.get("paths", {})
KB_PATH: str = os.path.join(BASE_DIR, PATHS_CONFIG.get("knowledge_base", "knowledge-base"))
VECTOR_STORE_PATH: str = os.path.join(BASE_DIR, PATHS_CONFIG.get("vector_store", "vector_store"))
METADATA_PATH: str = os.path.join(VECTOR_STORE_PATH, "metadata.pkl")

# ── Caching ─────────────────────────────────────────────────
CACHE_CONFIG: dict = _cfg.get("caching", {})

# ── EDGAR ───────────────────────────────────────────────────
EDGAR_CONFIG: dict = _cfg.get("edgar", {})

# ── Agentic RAG ────────────────────────────────────────────
AGENTIC_CONFIG: dict = _cfg.get("agentic", {})

# ── API keys (from .env) ───────────────────────────────────
PINECONE_API_KEY: str | None = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY: str | None = os.environ.get("TAVILY_API_KEY")

# ── AWS config (from .env) ─────────────────────────────────
AWS_DEFAULT_REGION: str = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
AWS_EMBEDDING_MODEL: str = os.environ.get("AWS_EMBEDDING_MODEL", "amazon.titan-embed-text-v1")
AWS_LLM_MODEL: str = os.environ.get("AWS_LLM_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")

# ── SEC EDGAR user agent (from .env) ───────────────────────
SEC_USER_AGENT: str = os.environ.get("SEC_USER_AGENT", "")
