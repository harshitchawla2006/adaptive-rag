from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from core.llm import fast_llm
from loguru import logger


class GradeDoc(BaseModel):
    "Binary score for doc relevance"
    score:str=Field(
        description="Is the document relevant to the queestion ? Answer 'yes or 'no"

    )

grader_llm=fast_llm.with_structured_output(GradeDoc)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a relevance grader. Given a retrieved document and a user question,
    decide if the document contains information useful to answer the question.
    Be strict — if the document is only loosely related, grade it as 'no'.
    Return 'yes' if relevant, 'no' if not."""),
    ("human", """Document:
{document}

Question: {question}

Is this document relevant?""")
])

grader_chain=prompt|grader_llm

def grade_documents(query:str,docs:list)->tuple[list,str]:
    """
    Grade each retrieved document.
    Returns filtered relevant docs and an overall grade verdict.
    """
    logger.info(f"Grading {len(docs)} docs for query:{query}")
    relevant_docs=[]

    for doc in docs:
        result=grader_chain.invoke({
            "document":doc.page_content,
            "question":query    
        })
        if result.score.lower()=="yes":
            relevant_docs.append(doc)
            logger.success(f"Doc graded :RELEVANT")

        else:
            logger.warning(f"Doc graded: NOT RELEVANT")

    if len(relevant_docs)>0:
        verdict="good"

    else:
        verdict="   poor"

    logger.info(f"Grade verdict: {verdict} ({len(relevant_docs)}/{len(docs)} relevant)")
    return relevant_docs, verdict



        