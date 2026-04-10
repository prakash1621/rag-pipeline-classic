import os
import pickle
import boto3
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings
from app.config import (
    VECTOR_STORE_PATH,
    METADATA_PATH,
    EMBEDDING_PROVIDER,
    EMBEDDING_CONFIG,
    VECTOR_STORE_PROVIDER,
    VECTOR_STORE_CONFIG,
    PINECONE_API_KEY,
    OPENAI_API_KEY,
    AWS_DEFAULT_REGION,
)

SUPPORTED_EMBEDDING_PROVIDERS = ["pinecone", "huggingface", "bedrock", "openai"]
SUPPORTED_VECTOR_STORE_PROVIDERS = ["faiss", "pinecone"]

# Embedding dimension per provider (used when auto-creating Pinecone indexes)
_EMBEDDING_DIMENSIONS = {
    "pinecone": {"multilingual-e5-large": 1024},
    "huggingface": {"all-MiniLM-L6-v2": 384},
    "bedrock": {"amazon.titan-embed-text-v1": 1536},
    "openai": {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072},
}


def _ensure_pinecone_index(index_name: str) -> None:
    """Create the Pinecone index if it doesn't already exist."""
    from pinecone import Pinecone, ServerlessSpec
    import time

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name in existing:
        return

    # Resolve dimension from config
    model_name = EMBEDDING_CONFIG.get("model_name", "")
    provider_dims = _EMBEDDING_DIMENSIONS.get(EMBEDDING_PROVIDER, {})
    dimension = provider_dims.get(model_name, 1024)

    cloud = VECTOR_STORE_CONFIG.get("cloud", "aws")
    region = VECTOR_STORE_CONFIG.get("region", "us-east-1")

    print(f"Creating Pinecone index '{index_name}' (dim={dimension}, cloud={cloud}, region={region})...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    # Wait until ready
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    print(f"Pinecone index '{index_name}' is ready.")


def get_embeddings():
    """Return a LangChain-compatible Embeddings object for the configured provider.

    Dispatches based on EMBEDDING_PROVIDER from config.yaml.
    Raises ValueError for unsupported providers.
    Raises EnvironmentError if a required API key is missing.
    """
    provider = EMBEDDING_PROVIDER

    if provider == "pinecone":
        if not PINECONE_API_KEY:
            raise EnvironmentError(
                "PINECONE_API_KEY is required when using the 'pinecone' embedding provider. "
                "Set it in your .env file."
            )
        from langchain_pinecone import PineconeEmbeddings

        model_name = EMBEDDING_CONFIG.get("model_name", "multilingual-e5-large")
        return PineconeEmbeddings(model=model_name)

    elif provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        model_name = EMBEDDING_CONFIG.get("model_name", "all-MiniLM-L6-v2")
        device = EMBEDDING_CONFIG.get("device", "cpu")
        return HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs={"device": device}
        )

    elif provider == "bedrock":
        region = EMBEDDING_CONFIG.get("region", AWS_DEFAULT_REGION)
        model_id = EMBEDDING_CONFIG.get("model_id", "amazon.titan-embed-text-v1")
        bedrock = boto3.client("bedrock-runtime", region_name=region)
        return BedrockEmbeddings(client=bedrock, model_id=model_id)

    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is required when using the 'openai' embedding provider. "
                "Set it in your .env file."
            )
        from langchain_openai import OpenAIEmbeddings

        model_name = EMBEDDING_CONFIG.get("model_name", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model_name)

    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. "
            f"Supported providers are: {SUPPORTED_EMBEDDING_PROVIDERS}"
        )

def create_vector_store(chunks, metadatas):
    """Create vector store from chunks. Dispatches on VECTOR_STORE_PROVIDER."""
    embeddings = get_embeddings()
    provider = VECTOR_STORE_PROVIDER

    if provider == "faiss":
        return FAISS.from_texts(chunks, embeddings, metadatas=metadatas)

    elif provider == "pinecone":
        from langchain_pinecone import PineconeVectorStore

        index_name = VECTOR_STORE_CONFIG.get("index_name")
        namespace = VECTOR_STORE_CONFIG.get("namespace")
        _ensure_pinecone_index(index_name)
        return PineconeVectorStore.from_texts(
            chunks,
            embeddings,
            metadatas=metadatas,
            index_name=index_name,
            namespace=namespace,
        )

    else:
        raise ValueError(
            f"Unsupported vector store provider: '{provider}'. "
            f"Supported providers are: {SUPPORTED_VECTOR_STORE_PROVIDERS}"
        )

def save_vector_store(vectorstore):
    """Persist vector store. FAISS: save_local(). Pinecone: no-op (cloud-persisted)."""
    provider = VECTOR_STORE_PROVIDER

    if provider == "faiss":
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        vectorstore.save_local(VECTOR_STORE_PATH)
    elif provider == "pinecone":
        # Pinecone is cloud-persisted; nothing to save locally.
        return
    else:
        raise ValueError(
            f"Unsupported vector store provider: '{provider}'. "
            f"Supported providers are: {SUPPORTED_VECTOR_STORE_PROVIDERS}"
        )

def load_vector_store():
    """Load existing vector store. FAISS: load_local(). Pinecone: connect to index."""
    provider = VECTOR_STORE_PROVIDER
    embeddings = get_embeddings()

    if provider == "faiss":
        if os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
            return FAISS.load_local(
                VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
            )
        return None

    elif provider == "pinecone":
        from langchain_pinecone import PineconeVectorStore

        index_name = VECTOR_STORE_CONFIG.get("index_name")
        namespace = VECTOR_STORE_CONFIG.get("namespace")
        _ensure_pinecone_index(index_name)
        return PineconeVectorStore.from_existing_index(
            index_name, embeddings, namespace=namespace
        )

    else:
        raise ValueError(
            f"Unsupported vector store provider: '{provider}'. "
            f"Supported providers are: {SUPPORTED_VECTOR_STORE_PROVIDERS}"
        )

def save_file_metadata(metadata):
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

def get_file_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'rb') as f:
            return pickle.load(f)
    return {}
