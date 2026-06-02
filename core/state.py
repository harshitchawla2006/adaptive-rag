from typing import List,TypedDict,Optional
from langchain_core.documents import Document

class GraphState(TypedDict):
    query:str
    retrieved_docs:Optional[List[Document]]
    grade:Optional[str]

    rewrite_count:int
    answer:Optional[str]
    eval_scores:Optional[dict]
