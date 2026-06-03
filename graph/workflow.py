from langgraph.graph import StateGraph, END
from core.state import GraphState
from graph.nodes import (
    retrieve_node,
    grade_node,
    web_search_node,
    generate_node,
    cache_check_node,
    cache_write_node
)
from graph.edges import (
    route_after_grading,
    route_after_web_search
)
from graph.query_enhancer import enhance_query
from graph.rewriter import rewrite_query
from loguru import logger


def route_after_cache(state: GraphState) -> str:
    """If cache hit, skip to end. Otherwise run full pipeline."""
    if state.get("answer"):
        logger.success("Cache hit → skipping to end")
        return "end"
    return "enhance_query"


def route_after_rewrite(state: GraphState) -> str:
    """After rewriting, always go back to retrieve."""
    rewrite_count = state["rewrite_count"]
    logger.info(f"--- EDGE: AFTER REWRITE --- count={rewrite_count}")
    return "retrieve"


def build_workflow():
    """Build and compile the LangGraph workflow."""

    workflow = StateGraph(GraphState)

    workflow.add_node("cache_check", cache_check_node)
    workflow.add_node("enhance_query", enhance_query)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generate", generate_node)
    workflow.add_node("cache_write", cache_write_node)
    workflow.set_entry_point("cache_check")
    workflow.add_conditional_edges(
        "cache_check",
        route_after_cache,
        {
            "end": END,
            "enhance_query": "enhance_query"
        }
    )

    workflow.add_edge("enhance_query", "retrieve")
    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        route_after_grading,
        {
            "generate": "generate",
            "web_search": "web_search"
        }
    )

    workflow.add_conditional_edges(
        "web_search",
        route_after_web_search,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )

    workflow.add_conditional_edges(
        "rewrite",
        route_after_rewrite,
        {
            "retrieve": "retrieve"
        }
    )

    workflow.add_edge("generate", "cache_write")
    workflow.add_edge("cache_write", END)

    app = workflow.compile()
    logger.success("Workflow compiled successfully")

    return app


rag_app = build_workflow()