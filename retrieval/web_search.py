from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv()

tavily_search=TavilySearch(
    max_results=5,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

def web_search(query:str)->list[Document]:
    """Search the web via Tavily and return results as langchain docs"""
    logger.info(f"Running Web Search for {query}")

    results=tavily_search.invoke(query)

    docs=[
        Document(
            page_content=result["content"],
            metadata={
                "source":result["url"],
                "type":"web_search"
            }
        )
        for result in results.get("results", [])
    ]

    logger.success(f"Web Search returned {len(docs)} results")
    return docs