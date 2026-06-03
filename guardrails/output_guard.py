from langchain_core.prompts import ChatPromptTemplate
from core.llm import fast_llm
from pydantic import BaseModel, Field
from loguru import logger

class OutputGuardResult(BaseModel):
    is_safe: bool = Field(description="Is the output safe to return?")
    reason: str = Field(description="Reason for the decision")

guard_llm=fast_llm.with_structured_output(OutputGuardResult)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a quality checker for a research assistant.
    Check if the generated answer is safe and appropriate to return.
    
    Mark as UNSAFE if the answer:
    - Contains harmful or dangerous information
    - Is completely unrelated to the question
    - Contains offensive content
    
    Mark as SAFE if the answer:
    - Directly addresses the question
    - Is factual and appropriate
    - Admits uncertainty honestly
    
    Return is_safe=true for safe answers, is_safe=false for unsafe ones."""),
    ("human", """Question: {question}
    
Answer: {answer}

Is this answer safe to return?""")
])

output_guard_chain = prompt | guard_llm

def check_output(query: str, answer: str) -> tuple[bool, str]:
    """
    Check if the generated answer is safe to return.
    Returns (is_safe, reason).
    """
    logger.info(f"--- GUARDRAIL: OUTPUT CHECK ---")
    result = output_guard_chain.invoke({
        "question": query,
        "answer": answer
    })

    if result.is_safe:
        logger.success(f"Output safe: {result.reason}")
    else:
        logger.warning(f"Output blocked: {result.reason}")

    return result.is_safe, result.reason

    