from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document
from loguru import logger
from typing import List
import os

def load_pdf(file_path:str)->List[Document]:
    """Load a PDF file and return a list of Document objects."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading PDF:{file_path}")
    loader=PyPDFLoader(file_path)
    docs=loader.load()
    logger.success(f"Loaded {len(docs)} pages from {file_path}")
    return docs

def load_web(url: str) -> List[Document]:
    """Load a web page and return a list of documents."""
    logger.info(f"Loading URL: {url}")
    loader = WebBaseLoader(url)
    docs = loader.load()
    logger.success(f"Loaded {len(docs)} documents from {url}")
    return docs

def load_documents(sources: List[str]) -> List[Document]:
    """
    Load from a mixed list of PDFs and URLs.
    Automatically detects type based on the source string.
    """
    all_docs = []

    for source in sources:
        if source.endswith(".pdf"):
            docs = load_pdf(source)
        elif source.startswith("http"):
            docs = load_web(source)
        else:
            logger.warning(f"Unknown source type, skipping: {source}")
            continue

        all_docs.extend(docs)

    logger.success(f"Total documents loaded: {len(all_docs)}")
    return all_docs


     