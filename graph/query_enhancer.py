from langchain_core.prompts import ChatPromptTemplate
from core.llm import fast_llm
from core.state import GraphState
from loguru import logger

hyde_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant. Given a question, write a 
    short hypothetical answer as if you already knew the answer.
    This will be used to improve document retrieval.
    Keep it under 100 words."""),
    ("human", "Question: {question}\n\nHypothetical answer:")
])

multi_query_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research assistant. Given a question, generate 
    3 different versions of it to retrieve more relevant documents.
    Each version should approach the topic from a different angle.
    Return ONLY the 3 questions, one per line, no numbering."""),
    ("human", "Original question: {question}\n\n3 alternative versions:")
])

hyde_chain = hyde_prompt | fast_llm
multi_query_chain = multi_query_prompt | fast_llm

def enhance_query(state:GraphState)->GraphState:
    logger.info(f"---Node query enhancer---")
    query=state["query"]

    hyde_response=hyde_chain.invoke({"question":query})
    hypothetical_answer=hyde_response.content
    logger.info(f"HYDE generated:{hypothetical_answer[:80]}..")

    multi_response = multi_query_chain.invoke({"question": query})
    alternative_queries = [
        q.strip()
        for q in multi_response.content.strip().split("\n")
        if q.strip()
    ]
    logger.info(f"Multi-query generated {len(alternative_queries)} alternatives")
    enhanced_queries = [query, hypothetical_answer] + alternative_queries
    return {**state,"enhanced_queries":enhanced_queries}
