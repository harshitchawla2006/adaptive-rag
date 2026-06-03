from core.state import GraphState
from loguru import logger

def route_after_grading(state:GraphState)->str:
    grade=state["grade"]
    rewrite_count=state["rewrite_count"]

    logger.info(f"--- EDGE: ROUTING --- grade={grade}, rewrites={rewrite_count}")
    if grade=="good":
        logger.success(f"Grade is good -> generating answer")
        return "generate"
    elif rewrite_count>=2:
        logger.warning(f"Max retries reached -> generating with what we have")
        return "generate"

    else:
        logger.warning("Grade is poor → falling back to web search")
        return "web_search"

def route_after_web_search(state: GraphState) -> str:
    """
    After web search, grade the results.
    If rewrite_count is maxed out, go straight to generate.
    """
    rewrite_count = state["rewrite_count"]

    if rewrite_count >= 2:
        logger.warning("Max rewrites reached → generating with what we have")
        return "generate"
    else:
        logger.info("Web search done → rewriting query for retry")
        return "rewrite"

