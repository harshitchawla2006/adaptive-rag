from langchain_core.prompts import ChatPromptTemplate
from core.llm import fast_llm
from pydantic import BaseModel, Field
from loguru import logger

class InputGuardResult(BaseModel):
    is_safe:bool=Field(description="Is the input safe to process?")
    reason:str=Field(description="Reason for the decision")

guard_llm=fast_llm.with_structured_output(InputGuardResult)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a safety checker for a research assistant.
    Your job is to block ONLY genuinely harmful requests.
    
    Mark as UNSAFE ONLY if the question:
    - Asks for illegal or dangerous information (weapons, drugs, etc.)
    - Contains clearly offensive or abusive content
    
    Mark as SAFE for everything else including:
    - Research and knowledge questions
    - Casual greetings like "hey" or "hello"
    - Vague or simple questions
    - Off-topic but harmless questions
    - Anything that is not clearly harmful
    
    When in doubt, mark as SAFE.
    Return is_safe=true for safe, is_safe=false for unsafe."""),
    ("human", "Question: {question}")
])
input_guard_chain = prompt | guard_llm

def check_input(query: str) -> tuple[bool, str]:
    """
    Check if the input query is safe to process.
    Returns (is_safe, reason).
    """
    logger.info(f"--- GUARDRAIL: INPUT CHECK ---")
    result = input_guard_chain.invoke({"question": query})
    
    if result.is_safe:
        logger.success(f"Input safe: {result.reason}")
    else:
        logger.warning(f"Input blocked: {result.reason}")
    
    return result.is_safe, result.reason

    
