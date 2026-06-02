from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger
from typing import List

def chunk_documents(docs:List[Document],chunk_size:int=512 ,chunk_overlap:int=100)->List[Document]:
    """Split Docs into chunks """
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n","\n",".","?","!"," ",""]
    )
    chunks=splitter.split_documents(docs)
    logger.info(f"Split {len(docs)} into {len(chunks)} chunks")
    return chunks   