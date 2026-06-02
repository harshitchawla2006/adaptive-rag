from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from retrieval.vectorstore import store_documents
from retrieval.hybrid import hybrid_search
from loguru import logger

def test_pipeline():
    # Step 1 — Load a real web page
    logger.info("=== STEP 1: Loading documents ===")
    sources = ["https://en.wikipedia.org/wiki/Retrieval-augmented_generation"]
    docs = load_documents(sources)

    # Step 2 — Chunk them
    logger.info("=== STEP 2: Chunking ===")
    chunks = chunk_documents(docs)

    # Step 3 — Store in Qdrant
    logger.info("=== STEP 3: Storing in Qdrant ===")
    store_documents(chunks)

    # Step 4 — Hybrid search
    logger.info("=== STEP 4: Hybrid search ===")
    results = hybrid_search("what is retrieval augmented generation", top_k=3)

    # Step 5 — Print results
    logger.info("=== RESULTS ===")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:300])
        print(f"Source: {doc.metadata.get('source', 'unknown')}")

if __name__ == "__main__":
    test_pipeline()