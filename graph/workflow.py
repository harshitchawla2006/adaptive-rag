from langgraph.graph import StateGraph, END
from core.state import GraphState
from graph.nodes import (
    retrieve_node,
    grade_node,
    web_search_node,
    generate_node
)
from graph.edges import (
    route_after_grading,
    route_after_web_search
)
from loguru import logger

def build_workflow():
    workflow=StateGraph(GraphState)

    workflow.add_node("retrieve",retrieve_node)
    workflow.add_node("grade",grade_node)
    workflow.add_node("web_search",web_search_node)
    workflow.add_node("generate",generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve","grade")
    workflow.add_conditional_edges(
        "grade",
        route_after_grading,
        {
            "generate": "generate",
            "web_search": "web_search"
        }
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    app=workflow.compile()
    logger.success(f"Workflow completed successfully")
    return app

rag_app = build_workflow()

    