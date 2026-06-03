from core.state import GraphState
from core.llm import smart_llm
from retrieval.hybrid import hybrid_search
from retrieval.grader import grade_documents
from retrieval.web_search import web_search
from langchain_core.prompts import ChatPromptTemplate
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



  

