from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from datasets import Dataset
from core.llm import fast_llm
from loguru import logger

ragas_llm = LangchainLLMWrapper(fast_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(
    FastEmbedEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5")
)

def get_score(results, key):
    """Safely extract a single float score from RAGAS results."""
    val = results[key]
    if isinstance(val, list):
        return round(float(val[0]), 3)
    return round(float(val), 3)

def evaluate_response(
    query: str,
    answer: str,
    retrieved_docs: list[Document]
):
    logger.info("-- Evaluating --")
    contexts = [doc.page_content for doc in retrieved_docs]
    data = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [answer]
    }
    dataset = Dataset.from_dict(data)
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    scores = {
        "faithfulness": get_score(results, "faithfulness"),
        "answer_relevancy": get_score(results, "answer_relevancy"),
        "context_recall": get_score(results, "context_recall"),
    }

    logger.success(f"Evaluation scores: {scores}")
    return scores