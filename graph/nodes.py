from core.state import GraphState
from core.llm import smart_llm
from retrieval.hybrid import hybrid_search
from retrieval.grader import grade_documents
from retrieval.web_search import web_search
from langchain_core.prompts import ChatPromptTemplate
from core.cache import check_cache, store_in_cache
from evaluation.ragas_eval import evaluate_response
from guardrails.input_guard import check_input
from guardrails.output_guard import check_output
from loguru import logger

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful research assistant. 
    Answer the question using ONLY the provided context.
    Be detailed, accurate and cite key points from the context.
    If the context doesn't have enough information, say so honestly."""),
    ("human", """Context:
{context}

Question: {question}

Provide a comprehensive answer:""")
])

generate_chain=prompt|smart_llm

def retrieve_node(state:GraphState)->GraphState:
    """node 1 hybrid search retrieval"""
    logger.info(f"---NODE RETRIEVE--")
    query=state["query"]
    docs=hybrid_search(query,top_k=5)
    return {**state,"retrieved_docs":docs}

def grade_node(state:GraphState)->GraphState:
    """node 2 CRAG grader"""
    logger.info(f"---NODE GRADE---")
    query=state["query"]
    docs=state["retrieved_docs"]
    relevant_docs,verdict=grade_documents(query,docs)
    return {**state,"retrieved_docs":relevant_docs,"grade":verdict}

def web_search_node(state:GraphState)->GraphState:
    """node 3 web search"""
    logger.info(f"---NODE WEB SEARCH---")
    query=state["query"]
    docs=state["retrieved_docs"]
    web_results=web_search(query)
    if docs is None:
        docs = []
    final_docs = docs + web_results
    return {**state,"retrieved_docs":final_docs}

def generate_node(state: GraphState) -> GraphState:
    """Node 4 — Generate final answer."""
    logger.info("--- NODE: GENERATE ---")
    query = state["query"]
    docs = state["retrieved_docs"]
    context = "\n\n".join([doc.page_content for doc in docs])

    response = generate_chain.invoke({
        "context": context,
        "question": query
    })

    answer = response.content
    logger.success(f"Answer generated: {answer[:100]}...")

    return {**state, "answer": answer}


def cache_check_node(state: GraphState) -> GraphState:
    """Node 0 — Check cache before running full pipeline."""
    logger.info("--- NODE: CACHE CHECK ---")
    query = state["query"]

    cached_answer = check_cache(query)

    if cached_answer:
        logger.success("Cache HIT — skipping full pipeline")
        return {**state, "answer": cached_answer, "grade": "cached"}
    
    logger.info("Cache MISS — running full pipeline")
    return {**state}


def cache_write_node(state: GraphState) -> GraphState:
    """Node 5 — Store answer in cache after generation."""
    logger.info("--- NODE: CACHE WRITE ---")
    query = state["query"]
    answer = state["answer"]
    eval_scores = state.get("eval_scores", None)

    store_in_cache(query, answer, eval_scores)

    return {**state}


def evaluate_node(state: GraphState) -> GraphState:
    """Node — Evaluate the generated answer using RAGAS."""
    logger.info("--- NODE: EVALUATE ---")
    
    query = state["query"]
    answer = state["answer"]
    docs = state["retrieved_docs"]

    # Skip evaluation if no docs (cache hit)
    if not docs:
        logger.info("Cache hit — skipping evaluation")
        return {**state}

    scores = evaluate_response(query, answer, docs)

    return {**state, "eval_scores": scores}
  
def input_guard_node(state: GraphState) -> GraphState:
    """Node — Check input safety before running pipeline."""
    logger.info("--- NODE: INPUT GUARD ---")
    query = state["query"]

    is_safe, reason = check_input(query)

    if not is_safe:
        return {
            **state,
            "answer": f"I can't help with that. {reason}",
            "grade": "blocked"
        }

    return {**state}


def output_guard_node(state: GraphState) -> GraphState:
    """Node — Check output safety before returning answer."""
    logger.info("--- NODE: OUTPUT GUARD ---")
    query = state["query"]
    answer = state["answer"]

    is_safe, reason = check_output(query, answer)

    if not is_safe:
        return {
            **state,
            "answer": f"I was unable to generate a safe response. {reason}"
        }

    return {**state}