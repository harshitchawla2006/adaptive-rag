from langchain_core.documents import Document
from retrieval.vectorstore import get_vectorstore
from rank_bm25 import BM25Okapi
from loguru import logger
from typing import List
import numpy as np

def reciprocal_rank_fusion(
    vector_results: List[Document],
    bm25_results: List[Document],
    k:int=60
)->List[Document]:
    """Combine sematic with bm25 search"""
    scores={}
    for rank,doc in enumerate(vector_results):
        key=doc.page_content
        if key not in scores:
            scores[key]={"doc":doc,"score":0.0}
        scores[key]["score"]+=(1/(rank+k))
    
    for rank, doc in enumerate(bm25_results):
        key = doc.page_content
        if key not in scores:
            scores[key] = {"doc": doc, "score": 0.0}
        scores[key]["score"] += 1 / (rank + k)

    sorted_results=sorted(
        scores.values(),
        key=lambda x:x["score"],
        reverse=True
    )
    return [item["doc"] for item in sorted_results]


def bm25_seacrh(
    query:str,
    all_docs:List[Document],
    top_k:int=5
)->List[Document]:
    """keyword match """
    tokenized_docs=[doc.page_content.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query=query.lower().split()
    scores=bm25.get_scores(tokenized_query)
    top_indices=np.argsort(scores)[::-1][:top_k]
    results=[all_docs[i] for i in top_indices]
    logger.info(f"BM25 retrieved {len(results)} documents")
    return results


def hybrid_search(query:str,top_k:int=5)->List[Document]:
    """Main retrieval fucntion composing of similarity search and bm25"""
    vectorstore=get_vectorstore()

    vector_results=vectorstore.similarity_search(query,k=top_k)
    logger.info(f"Vector search retrieved {len(vector_results)} documents")

    all_docs_raw,_=vectorstore.client.scroll(
        collection_name="research_docs",
        limit=1000,
        with_payload=True,
        with_vectors=False
    )

    all_docs=[
        Document(
            page_content=doc.payload.get("page_content",""),
            metadata=doc.payload.get("metadata",{})
        )
        for doc in all_docs_raw
    ]
    bm25_results=bm25_seacrh(query,all_docs,top_k=top_k)

    final_results=reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results,
        k=top_k
    )
    logger.info(f"Hybrid Search retrieved {len(final_results)} documents")  
    return final_results[:top_k]
    






