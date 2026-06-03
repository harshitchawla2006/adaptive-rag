
from langchain_core.prompts import ChatPromptTemplate
from core.llm import fast_llm
from core.state import GraphState
from loguru import logger

step_back_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at reformulating questions.
    Given a specific technical question, generate a broader version
    that captures the core concept being asked about.
    Be specific to the technical domain, not generic.
    Return ONLY the rewritten question, nothing else.
    
    Examples:
    Q: "What is the learning rate in Adam optimizer?"
    A: "How do deep learning optimizers control training speed?"
    
    Q: "What are LangGraph nodes?"
    A: "How does LangGraph structure AI agent workflows?"
    """),
    ("human", "Original question: {question}\n\nStep-back question:")
])

decompose_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at breaking down complex questions.
    Given a question, decompose it into 2-3 simpler sub-questions
    that together would answer the original question.
    Return ONLY the sub-questions, one per line, no numbering."""),
    ("human", "Complex question: {question}\n\nSub-questions:")
])

step_back_chain = step_back_prompt | fast_llm
decompose_chain = decompose_prompt | fast_llm

def rewrite_query(state:GraphState)->GraphState:
    logger.info("--- NODE: QUERY REWRITER ---")
    query = state["query"]
    rewrite_count = state["rewrite_count"]

    step_back_response=step_back_chain.invoke({"question"})
    step_back_query=step_back_response.content.strip()
    logger.info(f"step back query :{step_back_query}")

    decompose_response=decompose_chain.invoke({"question":query})
    sub_questions = [
        q.strip()
        for q in decompose_response.content.strip().split("\n")
        if q.strip()
    ]
    logger.info(f"Decomposed into {len(sub_questions)} sub-questions")

    new_query=step_back_query
    return {
        **state,
        "query":new_query,
        "rewrite_count":rewrite_count+1,
        "retrieved_docs":None,
        "grade":None
    }
    