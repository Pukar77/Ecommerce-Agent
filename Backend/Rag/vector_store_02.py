from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


def vector_stores(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    COLLECTION_NAME = "products_agentic_ai_new"

    # from_texts handles collection creation, embedding, and insertion
    # force_recreate=True ensures a clean slate every time ingestion runs
    vectorstore = QdrantVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        url="http://localhost:6333",
        collection_name=COLLECTION_NAME,
        force_recreate=True
    )

    # Verify insertion
    client = QdrantClient(url="http://localhost:6333")
    count = client.count(collection_name=COLLECTION_NAME)
    print(f"📦 Total vectors inserted: {count.count}")

    return vectorstore


