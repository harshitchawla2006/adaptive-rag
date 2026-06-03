from graph.workflow import rag_app
from loguru import logger


def run_agent(query: str):
    """Run the full CRAG agent on a query."""

    logger.info(f"Running agent for query: {query}")

    initial_state = {
        "query": query,
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
    print(f"DOCS USED: {len(final_state['retrieved_docs'])}")
    print(f"\nANSWER:\n{final_state['answer']}")
    print("="*60)

    return final_state


if __name__ == "__main__":
    run_agent("What is retrieval augmented generation?")

    run_agent("What is the latest version of LangGraph?")