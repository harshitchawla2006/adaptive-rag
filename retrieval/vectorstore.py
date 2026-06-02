from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from loguru import logger
from typing import List

embeddings=OllamaEmbeddings(model="nomic-embed-text")
client=QdrantClient(path="./qdrant_storage")
COLLECTION_NAME = "research_docs"
VECTOR_SIZE = 768

def get_or_create_collection():
    """Create Qdrabt collection """
    existing=[c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        logger.success(f"Created collection: {COLLECTION_NAME}")
    else:
        logger.info(f"Collection already exists: {COLLECTION_NAME}")

def store_documents(chunks:List[Document]):
    """Embed and store chunks"""
    get_or_create_collection()
    vectorstore=QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    vectorstore.add_documents(chunks)
    logger.success(f"Stored {len(chunks)} chunks in Qdrant")


def get_vectorstore()->QdrantVectorStore:
    """return vectorstore for retrieval"""
    get_or_create_collection()
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings    
    )
