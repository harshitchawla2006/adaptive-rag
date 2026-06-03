from graph.workflow import rag_app
from loguru import logger


def run_agent(query: str):
    """Run the full agent on a query."""
    logger.info(f"Running agent for: {query}")

    initial_state = {
        "query": query,
        "enhanced_queries": None,
        "retrieved_docs": None,
        "grade": None,
        "rewrite_count": 0,
        "answer": None,
        "eval_scores": None
    }

    final_state = rag_app.invoke(initial_state)

    print("\n" + "="*60)
    print(f"QUERY: {query}")
    print("="*60)
    print(f"GRADE: {final_state['grade']}")
    print(f"REWRITE COUNT: {final_state['rewrite_count']}")
    print(f"\nANSWER:\n{final_state['answer']}")
    print("="*60)

    return final_state


if __name__ == "__main__":
    print("\n>>> TEST 1 — First run (should miss cache)")
    run_agent("What is retrieval augmented generation?")

    print("\n>>> TEST 2 — Rephrased query (should hit cache)")
    run_agent("Can you explain what RAG is?")

    print("\n>>> TEST 3 — New question (should miss cache)")
    run_agent("What are the main components of a LangGraph workflow?")