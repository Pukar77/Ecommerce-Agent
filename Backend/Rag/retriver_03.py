from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


def _get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_vectorstore():
    """Return the QdrantVectorStore directly for flexible searches."""
    client = QdrantClient(url="http://localhost:6333")
    return QdrantVectorStore(
        client=client,
        collection_name="products_agentic_ai_new",
        embedding=_get_embeddings(),
        content_payload_key="page_content",
    )


def get_retrevial():
    retriever = get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    return retriever
