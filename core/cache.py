from langchain_ollama import OllamaEmbeddings
from loguru import logger
from datetime import datetime
import json
import os
import numpy as np

embeddings=OllamaEmbeddings(model="nomic-embed-text")

CACHE_FILE="./cache.json"

SIMILARITY_THRESHOLD=0.92

def load_cache() -> list:
    """Load cache from disk."""
    if not os.path.exists(CACHE_FILE):
        return []
    with open(CACHE_FILE, "r") as f:
        return json.load(f)

def save_cache(cache: list):
    """Save cache to disk."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def cosine_similarity(a:list,b:list)-> float:
    a,b=np.array(a),np.array(b)
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def check_cache(query:str)->str|None:
    cache=load_cache()
    if not cache:
        return None
    query_embedding=embeddings.embed_query(query)

    for entry in cache:
        similarity=cosine_similarity(query_embedding,entry["embedding"])
        logger.info(f"Cache sim: {similarity:.3f} for: {entry['query'][:50]}")
        if similarity>=SIMILARITY_THRESHOLD:
            logger.success(f"Cache HIT : Similarity={similarity:.3f}")
            return entry["answer"]

    logger.info(f"Cache MISS")
    return None


def store_in_cache(query:str,answer:str,eval_scores:dict=None):
    if eval_scores:
        faithfulness = eval_scores.get("faithfulness", 0)
        if faithfulness < 0.8:
            logger.warning(f"Answer quality too low to cache (faithfulness={faithfulness:.2f})")
            return

    cache = load_cache()

    query_embedding = embeddings.embed_query(query)

    entry = {
        "query": query,
        "answer": answer,
        "embedding": query_embedding,
        "timestamp": datetime.now().isoformat(),
        "eval_scores": eval_scores or {}
    }
    cache.append(entry)
    save_cache(cache)
    logger.success(f"Stored in cache: {query[:50]}...")
    





    